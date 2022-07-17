import h5py
import os
import glob
import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *

class BaseLoader(Dataset):
    def read_input(self, idx):
        pass

    def read_target(self, idx):
        pass

class FirstBreakLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = os.listdir(self.inputdir)
        self.targets = os.listdir(self.targetdir)
        self.transform = transform

    def __len__(self):
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        image = np.load(os.path.join(self.inputdir,self.inputs[idx]))
        return image[...,None]

    def read_target(self, idx):
        return np.load(os.path.join(self.targetdir,self.targets[idx]))[...,None]

    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample


class DerainLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = os.listdir(self.inputdir)
        self.targets = os.listdir(self.targetdir)
        self.transform = transform

    def __len__(self):
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        image = io.imread(os.path.join(self.inputdir,self.inputs[idx]))
        return image

    def read_target(self, idx):
        return io.imread(os.path.join(self.targetdir,self.targets[idx]))

    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample

def get_derain_dataset(rootdir="/home/makam0a/Dropbox/projects/denoising/Restormer/Deraining/Datasets/train/Rain13K",
                       min_size=256, crop_size=(224, 224), target_size=(224, 224), normalize=False,
                       noise_transforms=[]):
    transforms_ = []
    ImageNormalize = [InputNormalize(imagenet_mean, imagenet_std), TargetNormalize(imagenet_mean, imagenet_std)]
    ImageChangeType = [ChangeType(), Scale()]
    transforms_ += [MinResize(min_size=min_size)]
    transforms_ += ImageChangeType
    transforms_ += [RandomCrop(crop_size)]
    if crop_size != target_size:
        transforms_ += [Resize(target_size)]
    transforms_ += [FlipChannels(), ToTensor()]
    if normalize:
        transforms_ += ImageNormalize
    transforms_ += noise_transforms
    return DerainLoader(rootdir, transform=transforms.Compose(transforms_))

def get_first_break_dataset(rootdir="/home/makam0a/Dropbox/gendata/data/",
                       noise_transforms=[]):
    transforms_ = []
    transforms_ += [FlipChannels(), ToTensor()]
    transforms_ += noise_transforms
    return FirstBreakLoader(rootdir, transform=transforms.Compose(transforms_))

def get_dataset(dtype, *pargs, **kwargs):
    if dtype == 'derain':
        dataset = get_derain_dataset(*pargs, **kwargs)
    elif dtype == 'firstbreak':
        dataset = get_first_break_dataset()
    else:
        raise ValueError("Unknown Dataset Type")
    return dataset