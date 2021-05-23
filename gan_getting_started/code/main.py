'''
@file: main.py

@author: rukman.sai@gmail.com
@created: April 10th 2021
'''

import os
import logging
import random
import argparse
import itertools
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GeneratorNetwork, DiscriminatorNetwork
from options import get_parser
from train_utils import HistoryBuffer
from data import ImageDataset
from train_utils import get_lr_lambda


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


class Learner:
    '''The class for performing the learning'''

    def __init__(self, args: argparse.Namespace) -> None:
        '''Initialize the instance

        Parameters
        ----------
        args
            The command line arguments
        '''
        # define the models
        self.args = args
        self.device = torch.device(
            'cuda' if (not args.cpu and torch.cuda.is_available()) else 'cpu'
        )
        logging.info(f'We are using {self.device} as the computation device')
        self.G = GeneratorNetwork(args).to(self.device)
        self.F = GeneratorNetwork(args).to(self.device)
        self.D_X = DiscriminatorNetwork(args).to(self.device)
        self.D_Y = DiscriminatorNetwork(args).to(self.device)
        self.history_X = HistoryBuffer(args)  # for D_X update
        self.history_Y = HistoryBuffer(args)  # for D_Y update

        self.optimizer_gen = torch.optim.Adam(
            itertools.chain(self.G.parameters(), self.F.parameters()),
            lr=args.lr, betas=(0.5, 0.999)
        )
        self.optimizer_disc = torch.optim.Adam(
            itertools.chain(self.D_X.parameters(), self.D_Y.parameters()),
            lr=args.lr, betas=(0.5, 0.999)
        )

        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_gen, get_lr_lambda(args)
        )
        self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_disc, get_lr_lambda(args)
        )

        # state keeping
        self._num_steps = 0  # total no. of global steps done
        self._epochs = 0  # total epochs done

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epoch: int):
        '''Setter for epochs property'''
        self._epochs = epoch

    def train(self, mode: bool = True) -> None:
        '''Set the learner in train or eval mode

        Parameters
        ----------
        mode
            Whether to set the learner in train or eval mode
        '''
        if mode:
            self.G.train()
            self.F.train()
            self.D_X.train()
            self.D_Y.train()
        else:
            self.G.eval()
            self.F.eval()
            self.D_X.eval()
            self.D_Y.eval()

    def eval(self,) -> None:
        '''Set the learner in eval mode'''
        self.train(False)

    def train_step(self, X, Y) -> Dict[str, Any]:
        '''Perform a single training step

        Parameters
        ----------
        X
            The input samples from dataset X
        Y
            The input samples from dataset Y

        Returns
        -------
        Dict[str, Any]
            The output to be logged to tensorboard
        '''

        X = X.to(self.device)
        Y = Y.to(self.device)

        # forward propagation
        Y_gen = self.G(X)
        Y_gen_prob = self.D_Y(Y_gen)
        Y_gen_prob_detach = self.D_Y(self.history_Y.get_sample(Y_gen).detach())
        Y_prob = self.D_Y(Y)
        X_identity = self.F(Y_gen)

        X_gen = self.F(Y)
        X_gen_prob = self.D_X(X_gen)
        X_gen_prob_detach = self.D_X(self.history_X.get_sample(X_gen).detach())
        X_prob = self.D_X(X)
        Y_identity = self.G(X_gen)

        # loss calculation
        G_target = torch.ones_like(Y_gen_prob)
        F_target = torch.ones_like(X_gen_prob)
        generator_loss = (
            F.mse_loss(Y_gen_prob, G_target, reduction='mean') +
            F.l1_loss(X_identity, X, reduction='mean') +
            F.mse_loss(X_gen_prob, F_target, reduction='mean') +
            F.l1_loss(Y_identity, Y, reduction='mean')
        )

        D_Y_fake_target = torch.zeros_like(Y_gen_prob_detach)
        D_Y_real_target = torch.ones_like(Y_prob)
        D_Y_loss = 0.5 * (
            F.mse_loss(Y_gen_prob_detach, D_Y_fake_target, reduction='mean') +
            F.mse_loss(Y_prob, D_Y_real_target, reduction='mean')
        )

        D_X_fake_target = torch.zeros_like(X_gen_prob_detach)
        D_X_real_target = torch.ones_like(X_prob)
        D_X_loss = 0.5 * (
            F.mse_loss(X_gen_prob_detach, D_X_fake_target, reduction='mean') +
            F.mse_loss(X_prob, D_X_real_target, reduction='mean')
        )

        # backward propagation
        self.optimizer_gen.zero_grad()
        self.disc_set_requires_grad(False)  # don't save grads during backprop
        generator_loss.backward()
        self.optimizer_gen.step()

        self.optimizer_disc.zero_grad()
        self.disc_set_requires_grad(True)
        D_Y_loss.backward()
        D_X_loss.backward()
        self.optimizer_disc.step()

        self._num_steps += 1

        logging_output = {
            "generator_loss": generator_loss.item(),
            "discriminatorX_loss": D_X_loss.item(),
            "discriminatorY_loss": D_Y_loss.item(),
            "photo_to_monet": torch.cat((X, Y_gen), dim=-1),
            "monet_to_photo": torch.cat((Y, X_gen), dim=-1),
        }

        return logging_output

    def disc_set_requires_grad(self, requires_grad: bool) -> None:
        '''Set the requires_grad for parameters in the discriminator

        By setting the requires grad option to False, one can discard the
        gradients calculated for those layers - this doesn't affect the
        backpropagation flow. The layers still participate in backpropagation
        but discard the gradients calculated for themselves.

        Parameters
        ----------
        requires_grad
            Whether the parameters require gradients or not
        '''
        for param in itertools.chain(self.D_X.parameters(),
                                     self.D_Y.parameters()):
            param.requires_grad = requires_grad

    def lr_step(self,) -> None:
        '''Take a lr scheduler step'''
        self.gen_scheduler.step()
        self.disc_scheduler.step()

    def save_checkpoint(self,) -> None:
        '''Save the checkpoint'''
        state_dict = {
            'G': self.G.state_dict(),
            'F': self.F.state_dict(),
            'D_X': self.D_X.state_dict(),
            'D_Y': self.D_Y.state_dict(),
            'optimizer_gen': self.optimizer_gen.state_dict(),
            'optimizer_disc': self.optimizer_disc.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict(),
            'disc_scheduler': self.disc_scheduler.state_dict(),
            'num_steps': self._num_steps,
            'epochs': self._epochs
        }
        torch.save(state_dict,
                   os.path.join(self.args.model_dir, 'checkpoint_last.pt'))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        '''Load the checkpoint

        Parameters
        ----------
        checkpoint_path
            The path to the checkpoint
        '''
        logging.info(f'Loading Checkpoint from path: {checkpoint_path}')

        # load checkpoint on cpu
        cpu_device = torch.device('cpu')
        state_dict = torch.load(checkpoint_path, cpu_device)

        # load the state dicts
        self.G.load_state_dict(state_dict['G'])
        self.G.to(self.device)
        self.F.load_state_dict(state_dict['F'])
        self.F.to(self.device)
        self.D_X.load_state_dict(state_dict['D_X'])
        self.D_X.to(self.device)
        self.D_Y.load_state_dict(state_dict['D_Y'])
        self.D_Y.to(self.device)

        if not self.args.reset_optim:  # load optimization state
            self.optimizer_gen.load_state_dict(state_dict['optimizer_gen'])
            self.optimizer_disc.load_state_dict(state_dict['optimizer_disc'])
            self.gen_scheduler.load_state_dict(state_dict['gen_scheduler'])
            self.disc_scheduler.load_state_dict(state_dict['disc_scheduler'])
            self._num_steps = state_dict['num_steps']
            self._epochs = state_dict['epochs']


def main(args: argparse.Namespace):
    '''The main function performing the training

    Parameters
    ----------
    args
        The command line arguments
    '''
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    # set the seed for replicating results
    random.seed(args.python_seed)
    torch.random.manual_seed(args.torch_seed)

    # initialize the learner
    learner = Learner(args)

    # load checkpoint if it exists
    if not args.finetune_model:  # finetune path not given
        default_path = os.path.join(args.model_dir, 'checkpoint_last.pt')
        if os.path.isfile(default_path):  # Resuming training
            learner.load_checkpoint(default_path)
    else:
        learner.load_checkpoint(args.finetune_model)

    # initialize the datasets and data loaders
    photos_dataset = ImageDataset(args, 'photo_jpg')

    monet_dataset = ImageDataset(args, 'monet_jpg')
    monet_sampler = torch.utils.data.RandomSampler(
        monet_dataset, replacement=True, num_samples=len(photos_dataset)
    )

    monet_data_loader = torch.utils.data.DataLoader(
        monet_dataset,
        batch_size=args.batch_size,
        sampler=monet_sampler,
        num_workers=args.data_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        persistent_workers=(True if args.data_workers > 0 else False),
    )

    photos_data_loader = torch.utils.data.DataLoader(
        photos_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.data_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        persistent_workers=(True if args.data_workers > 0 else False),
    )

    with SummaryWriter() as writer:
        learner.train()  # set in train mode

        with tqdm(desc='Epoch', total=args.epochs) as epochbar:
            for epoch in range(learner.epochs + 1, args.epochs + 1):
                with tqdm(desc='Step', total=len(photos_dataset)) as stepbar:
                    # update tqdm to the required num_steps -
                    # useful when we load a checkpoint
                    stepbar.update(learner.num_steps % len(photos_dataset))

                    for X, Y in zip(photos_data_loader, monet_data_loader):
                        # one train-step
                        logging_output = learner.train_step(X, Y)

                        # progress bar update and logging
                        stepbar.update()
                        if (
                            not (learner.num_steps % args.log_steps) or
                                not (learner.num_steps % len(photos_dataset))
                        ):
                            writer.add_scalar(
                                'Loss/generator_loss',
                                logging_output['generator_loss'],
                                learner.num_steps
                            )
                            writer.add_scalar(
                                'Loss/discriminatorX_loss',
                                logging_output['discriminatorY_loss'],
                                learner.num_steps
                            )
                            writer.add_scalar(
                                'Loss/discriminatorY_lossY',
                                logging_output['discriminatorX_loss'],
                                learner.num_steps
                            )
                            writer.add_images(
                                'Translation/photo_to_monet',
                                logging_output['photo_to_monet'],
                                learner.num_steps
                            )
                            writer.add_images(
                                'Translation/monet_to_photo',
                                logging_output['monet_to_photo'],
                                learner.num_steps
                            )

                # epoch ended
                learner.lr_step()
                learner.epochs = epoch
                epochbar.update(learner.epochs - epochbar.n + 1)

                # save checkpoint
                tqdm.write('Saving Checkpoint...')
                learner.save_checkpoint()


if __name__ == '__main__':
    parser = get_parser()

    args = parser.parse_args()

    print(args)

    main(args)
