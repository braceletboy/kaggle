'''
@file: options.py

@author: rukman.sai@gmail.com
@created: April 10th 2021
'''

import os
import argparse


def check_datadir_structure(path: str):
    '''Check if the given data directory is of the right structure.

    It is expected that the data directory is of the following structure:

    ROOTDIR/
        -   monet_jpg/
            -   monet_photo1.jpg
            -   monet_photo2.jpg
            -   monet_photo3.jpg
            -   ...
            -   ...

        -   photo_jpg/
            -   natural_photo1.jpg
            -   natural_photo2.jpg
            -   natural_photo3.jpg
            -   ...
            -   ...

    Parameters
    ----------
    path
        The path to the root of the data directory
    '''
    monet_dir = os.path.join(path, 'monet_jpg')
    photos_dir = os.path.join(path, 'photo_jpg')
    if os.path.isdir(monet_dir) and os.path.isdir(photos_dir):
        return path
    else:
        raise IOError('The directory structure of data directory is wrong.')


def get_parser():
    parser = argparse.ArgumentParser()

    # Data Loading
    parser.add_argument('--root_datadir', type=check_datadir_structure,
                        default='../data', help='The path to the root of the '
                        'dataset directory')
    parser.add_argument('--data_workers', type=int,
                        default=os.cpu_count(), help='The number of worker '
                        'for data loading. If this is >1, it means that we '
                        'will be performing multi-process data loading')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--shuffle', action='store_true', help='Whether to '
                        'shuffle the dataset between epochs')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to '
                        'use pinned memory while data loading')
    parser.add_argument('--drop_last', action='store_true', help='Whether to '
                        'drop the last incomplete batch')

    # Model Definition
    parser.add_argument('--residual_blocks', type=int, default=9, help='The '
                        'number of residual blocks in the generator')

    # Training
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead '
                        'of GPU even if GPU is available')
    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-4,
                        help='The base learing rate to be used for training')
    parser.add_argument('--epochs', type=int, default=200, help='The total '
                        'number of training epochs')
    parser.add_argument('--decay_epochs', type=int, default=100, help='The '
                        'number of epochs for which we will decay the lr')
    parser.add_argument('--history_size', type=int, default=50, help='The '
                        'size of the history of generated images to be '
                        'maintained for discriminator update')
    parser.add_argument('--python_seed', type=int, default=20, help='The '
                        'randomization seed for python')
    parser.add_argument('--torch_seed', type=int, default=10, help='The '
                        'randomization seed for torch. Torch doesn\'t use '
                        'python seed')
    parser.add_argument('--log_steps', type=int, default=100, help='')

    # model-checkpointing
    parser.add_argument('--model_dir', type=str, default='./model_checkpoints',
                        help='The directory where we will store our model '
                        'checkpoints')
    parser.add_argument('--finetune_model', type=str, default=None,
                        help='The path to the model checkpoint to finetune')
    parser.add_argument('--reset_optim', action='store_true', help='Whether '
                        'to reset the optimization state before finetuning')

    return parser
