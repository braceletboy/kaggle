'''
@file: data.py

@author: rukman.sai@gmail.com
@created: April 11th 2020
'''

import os
from glob import glob
from argparse import Namespace
from typing import Tuple

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    '''The dataset that loads images'''

    def __init__(
        self,
        args: Namespace,
        image_dirname: str,
        input_img_size: Tuple[int, int] = (256, 256),
    ):
        '''Initialize the instance

        Parameters
        ----------
        args
            Command line arguments
        image_dirname
            The name of the directory that contains the image dataset. This can
            be either 'photo_jpg' or 'monet_jpg'
        input_img_size
            The input images will be resized to this size
        '''
        assert image_dirname == 'photo_jpg' or image_dirname == 'monet_jpg', (
            "image_dirname can either be 'photo_jpg' or 'monet_jpg' but "
            f"'{image_dirname}' is given"
        )
        self.datadir = os.path.join(args.root_datadir, image_dirname)
        self.file_list = glob(os.path.join(self.datadir, '*.jpg'))
        self.resize_transform = torchvision.transforms.Resize(input_img_size)
        self.totensor_transform = torchvision.transforms.ToTensor()

    def __getitem__(self, idx: int) -> torch.Tensor:
        '''Get the item with the given index

        Parameters
        ----------
        idx
            The index of the sample we want to fetch/get

        Returns
        -------
        torch.Tensor
            The image in tensor format
        '''
        filepath = self.file_list[idx]
        pil_image = Image.open(filepath)
        input = self.totensor_transform(pil_image)
        input = self.resize_transform(input)
        return input

    def __len__(self,):
        '''Return the number of samples in the dataset'''
        return len(self.file_list)
