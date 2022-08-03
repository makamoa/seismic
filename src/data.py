import os
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *
import yaml

# Read YAML file
with open(os.path.join("../config/", "global_config.yml"), 'r') as stream:
    data_loaded = yaml.safe_load(stream)
SEISMICROOT = data_loaded['SEISMICROOT']
DERAINROOT = data_loaded['DERAINROOT']
DERAINTRAIN = os.path.join(SEISMICROOT, 'train/Rain13K')
DERAINTEST = os.path.join(SEISMICROOT, 'test/Rain13K')
SEISMICDIR = os.path.join(SEISMICROOT, 'data/')


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
        self.class_names = ['empty', 'wave']

    def __len__(self):
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        image = np.load(os.path.join(self.inputdir,self.inputs[idx]))
        return image[...,None]

    def read_target(self, idx):
        return np.load(os.path.join(self.targetdir,self.targets[idx]))

    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample


class DenoiseLoader(FirstBreakLoader):
    def __init__(self, *pargs, **kwargs):
        super(DenoiseLoader, self).__init__(*pargs, **kwargs)
        self.targetdir = self.inputdir
        self.targets = self.inputs
        self.transform = self.transform

    def read_target(self, idx):
        return super(DenoiseLoader, self).read_target(idx)[...,None]



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

def get_derain_dataset(rootdir=DERAINTRAIN,
                       min_size=256, crop_size=(224, 224), target_size=(224, 224), normalize=False,
                       noise_transforms=[]):
    transforms_ = []
    ImageNormalize = [InputNormalize(imagenet_mean, imagenet_std)]
    ImageChangeType = [ChangeType(), Scale()]
    transforms_ += [MinResize(min_size=min_size)]
    transforms_ += ImageChangeType
    transforms_ += [RandomCrop(crop_size)]
    if crop_size != target_size:
        transforms_ += [Resize(target_size)]
    transforms_ += noise_transforms
    transforms_ += [FlipChannels(), ToTensor()]
    if normalize:
        transforms_ += ImageNormalize
    return DerainLoader(rootdir, transform=transforms.Compose(transforms_))

def get_first_break_dataset(rootdir=SEISMICDIR,
                            target_size=(224, 224),
                            noise_transforms=[]):
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType(problem='segment')]
    transforms_ += [ScaleNormalize('input')]
    transforms_ += [FlipChannels(only_input=True), ToTensor()]
    return FirstBreakLoader(rootdir, transform=transforms.Compose(transforms_))

def get_denoise_dataset(rootdir=SEISMICDIR,
                       noise_transforms=[]):
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType()]
    transforms_ += [ScaleNormalize('input'), ScaleNormalize('target')]
    transforms_ += [FlipChannels(), ToTensor()]
    return DenoiseLoader(rootdir, transform=transforms.Compose(transforms_))

def get_dataset(dtype, *pargs, **kwargs):
    if dtype == 'derain':
        dataset = get_derain_dataset(*pargs, **kwargs)
    elif dtype == 'firstbreak':
        dataset = get_first_break_dataset(*pargs, **kwargs)
    elif dtype == 'denoise':
        dataset = get_denoise_dataset(*pargs, **kwargs)
    else:
        raise ValueError("Unknown Dataset Type")
    return dataset

def get_train_val_dataset(dataset, valid_split=0.1):
    train_size = int((1 - valid_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, val_dataset