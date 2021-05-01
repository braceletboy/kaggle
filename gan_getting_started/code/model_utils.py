'''
@file: model_utils.py

@author: rukman.sai@gmail.com
@created: April 10th 2021
'''

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''One Residual Block'''

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sz: int = 3,
        stride: float = 1,
        instance_norm: bool = False,
        final_relu: bool = False,
    ):
        '''Initialize the instance

        in_channels
            The number of channels in the input
        out_channels
            The number of channels in the output
        kernel_sz
            The convolution kernel size
        stride
            The stride to use in the first convolution operation
        instance_norm
            Whether to use instance norm or not
        final_relu
            Whether to use relu after the addition operation. The original
            Resnet Block used this but the following ablation study:

            http://torch.ch/blog/2016/02/04/resnets.html

            shows that if this final relu layer is not used, there can be a
            slight increase in test performance
        '''
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(
            in_channels, out_channels, kernel_sz, stride, relu_type='relu',
            padding_type='reflect', instance_norm=instance_norm
        )
        self.conv_block2 = ConvBlock(
            out_channels, out_channels, kernel_sz,
            relu_type=('relu' if final_relu else 'none'),
            padding_type='reflect', instance_norm=instance_norm
        )

    def forward(self, input: torch.Tensor):
        '''Forward Propagation'''
        residual = input
        input = self.conv_block1(input)
        input = self.conv_block2(input, residual)
        return input


class ConvBlock(nn.Module):
    '''One Convolution layer with normalization and non-linear units'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        stride: float = 1,
        relu_type: str = 'none',
        padding_type: str = 'none',
        transposed_conv: bool = False,
        instance_norm: bool = False,
    ):
        '''Iniatialize the instance

        Parameters
        ----------
        in_channels
            The number of input channels
        out_channels
            The number of output channels
        kernel_sz
            The convolution kernel size
        stride
            The convolution stride
        transposed_conv
            Whether to use transposed convolution or the normal one
        instance_norm
            Whether to use instance norm or batch norm
        relu_type
            The relu type to use - possible values: 'relu', 'leakyrelu', 'none'
        padding_type
            The padding to use - possible values: 'reflect', 'zero', 'none'
        '''

        super(ConvBlock, self).__init__()
        relu_type = relu_type.lower()
        padding_type = padding_type.lower()

        # choose convolution
        if transposed_conv:
            ConvX = nn.ConvTranspose2d
        else:
            ConvX = nn.Conv2d

        self.padding_layer = None

        # choose padding
        if padding_type == 'reflect':
            self.padding_layer = nn.ReflectionPad2d(int((kernel_sz-1)/2))
            self.conv_layer = ConvX(in_channels, out_channels,
                                    kernel_sz, stride, padding=0)
        elif padding_type == 'zero':
            kwargs = {'padding': int((kernel_sz-1)/2)}
            if transposed_conv:
                kwargs['output_padding'] = int((kernel_sz-1)/2)
            self.conv_layer = ConvX(in_channels, out_channels,
                                    kernel_sz, stride, **kwargs)
        elif padding_type == 'none':
            self.conv_layer = ConvX(in_channels, out_channels,
                                    kernel_sz, stride, padding=0)
        else:
            raise ValueError(
                "'padding_type' can only take 'reflect', 'leakdyrelu', 'none' "
                f"values but '{padding_type} was given'"
            )

        # choose normalization
        if instance_norm:
            self.norm_layer = nn.InstanceNorm2d(out_channels)
        else:
            self.norm_layer = nn.BatchNorm2d(out_channels)

        # choose relu
        if relu_type == 'leakyrelu':
            self.relu_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif relu_type == 'relu':
            self.relu_layer = nn.ReLU(inplace=True)
        elif relu_type == 'none':
            self.relu_layer = None
        else:
            raise ValueError(
                "'relu_type' can only take 'relu', 'leakyrelu', 'none' values"
                f" but '{relu_type}' was given"
            )

    def forward(self, input: torch.Tensor, residual: torch.Tensor = None):
        '''Forward Propagation

        Parameters
        ----------
        input
            The input to this neural net block during forward prop
        residual
            The residual term that needs to be added to the input - used in
            residual blocks
        '''
        if self.padding_layer:
            input = self.padding_layer(input)
        input = self.conv_layer(input)
        input = self.norm_layer(input)
        if residual is not None:
            input = input + residual
        if self.relu_layer:
            input = self.relu_layer(input)
        return input
