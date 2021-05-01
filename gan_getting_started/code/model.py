'''
@file: model.py

@author: rukman.sai@gmail.com
@created: April 10th 2021
'''

from argparse import Namespace

import torch
import torch.nn as nn

from model_utils import ResidualBlock, ConvBlock


class GeneratorNetwork(nn.Module):
    '''The Generator Network'''

    def __init__(self, args: Namespace):
        '''Initialize the instance

        Parameters
        ----------
        args
            Command line arguments
        '''
        super(GeneratorNetwork, self).__init__()
        self.conv_block_first = ConvBlock(3, 64, 7, 1,
                                          relu_type='relu',
                                          padding_type='reflect',
                                          instance_norm=True)
        self.down_block1 = ConvBlock(64, 128, 3, 2,
                                     relu_type='relu',
                                     padding_type='zero',
                                     instance_norm=True)
        self.down_block2 = ConvBlock(128, 256, 3, 2,
                                     relu_type='relu',
                                     padding_type='zero',
                                     instance_norm=True)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(256, 256, 3, 1, instance_norm=True)
                for _ in range(args.residual_blocks)
            ]
        )
        self.up_block1 = ConvBlock(256, 128, 3, 2,
                                   relu_type='relu',
                                   padding_type='zero',
                                   transposed_conv=True,
                                   instance_norm=True)
        self.up_block2 = ConvBlock(128, 64, 3, 2,
                                   relu_type='relu',
                                   padding_type='zero',
                                   transposed_conv=True,
                                   instance_norm=True)
        self.conv_block_last = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor):
        '''Forward Propagation'''
        input = self.conv_block_first(input)
        input = self.down_block1(input)
        input = self.down_block2(input)
        input = self.residual_blocks(input)
        input = self.up_block1(input)
        input = self.up_block2(input)
        input = self.conv_block_last(input)
        return input


class DiscriminatorNetwork(nn.Module):
    '''The Discriminator Network

    We use a 70x70 Patch GAN discriminator. Here the 70 refers to the receptive
    field of our discriminator network. One can read more about receptive
    fields here:

    https://distill.pub/2019/computing-receptive-fields/
    '''

    def __init__(self, args: Namespace):
        '''Initialize the instance

        Parameters
        ----------
        args
            Command line arguments
        '''
        super(DiscriminatorNetwork, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_block2 = ConvBlock(64, 128, 4, 2, relu_type='leakyrelu',
                                     padding_type='zero', instance_norm=True)
        self.conv_block3 = ConvBlock(128, 256, 4, 2, relu_type='leakyrelu',
                                     padding_type='zero', instance_norm=True)
        self.conv_block4 = ConvBlock(256, 512, 4, 1, relu_type='leakyrelu',
                                     padding_type='zero', instance_norm=True)
        self.conv_layer_last = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, input: torch.Tensor):
        '''Forward Propagation'''
        input = self.conv_block1(input)
        input = self.conv_block2(input)
        input = self.conv_block3(input)
        input = self.conv_block4(input)
        input = self.conv_layer_last(input)
        return input
