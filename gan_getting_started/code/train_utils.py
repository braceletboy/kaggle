'''
@file: linear_decay_scheduler.py

@author: rukman.sai@gmail.com
@created: April 22nd 2021
'''

import random
import argparse
from typing import Callable

import torch


def get_lr_lambda(args: argparse.Namespace) -> Callable:
    '''Return the linear decay learning rate function

    This is a closure. It is defined so that one can configure the linear
    decay function based on the command line arguments passed

    Parameters
    ----------
    args
        The command line arguments

    Returns
    -------
    Callable
        The function for getting a linearly decaying learning rate
    '''
    base_lr = args.lr
    total_epochs = args.epochs
    decay_epochs = args.decay_epochs
    plateau_epochs = total_epochs - decay_epochs

    assert plateau_epochs >= 0, (
        f'Decay epochs: {decay_epochs} exceeds '
        f'Total epochs: {total_epochs}'
    )

    def linear_decay_lambda(epoch: int):
        '''Decay the learning rate linearly based on the epoch

        Parameters
        ----------
        epoch
            The current epoch
        '''
        if epoch < plateau_epochs:
            return base_lr
        else:
            return base_lr * (1 - (epoch - plateau_epochs)/decay_epochs)

    return linear_decay_lambda


class HistoryBuffer:
    '''Class for maintaining a history of images for discriminator update'''

    def __init__(
        self,
        args: argparse.Namespace
    ) -> None:
        '''Initialize the instance

        Parameters
        ----------
        args
            The command line arguments
        '''
        self.buffer_size = args.history_size
        self._buffer_samples = []

    def get_sample(
        self,
        current_generated_sample: torch.Tensor
    ) -> torch.Tensor:
        '''Return images from the buffer for discriminator update

        Parameters
        ----------
        current_generated_sample
            The generated sample for the current training step

        Returns
        -------
        torch.Tensor
            The images used for discriminator update step
        '''
        if self.buffer_size == 0:
            return current_generated_sample

        update_images = []
        for image in current_generated_sample:
            image = torch.unsqueeze(image, dim=0)
            if len(self._buffer_samples) < self.buffer_size:
                self._buffer_samples.append(image)
                update_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # image from history - 50% probability
                    random_idx = random.randint(0, self.buffer_size - 1)
                    up_img = self._buffer_samples[random_idx].clone()
                    self._buffer_samples[random_idx] = image
                    update_images.append(up_img)
                else:  # same image - 50% probability
                    update_images.append(image)

        return torch.cat(update_images, dim=0)
