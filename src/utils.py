import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms
from skimage.transform import AffineTransform, warp

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        x = sample['input']
        y = sample['target']
        if np.random.rand(1) < self.prob:
            sample['input'] = torch.flip(x, dims=(-1,))
            sample['target'] = torch.flip(y, dims=(-1,))
        return sample

class ScaleNormalize():
    def __init__(self, type='input'):
        self.type = type
    def __call__(self, sample):
        x = sample[self.type]
        x -= x.min()
        x /= x.max()
        x *= 2
        x -= 1
        return sample

class RandomShift():
    def __init__(self, mean=1, std=1e-5, shift_mean=0, shift_std=1e-3):
        self.mean = mean
        self.std = std
        self.shift_mean = shift_mean
        self.shift_std = shift_std

    def __call__(self, sample):
        image, target = sample['input'], sample['target']
        wx, wy = image.shape[:2]
        scale = np.random.uniform(low=self.mean - np.sqrt(3) * self.std, high=self.mean + np.sqrt(3) * self.std)
        shift = scale * np.random.uniform(low=self.shift_mean - np.sqrt(3) * self.shift_std, high=self.shift_mean + np.sqrt(3) * self.shift_std)
        scaling = AffineTransform(scale=(1, scale), translation=(0, wy - scale*wy - 1))
        scaled = warp(image, scaling, mode='constant', preserve_range=True)
        scaled_target = warp(target, scaling, mode='constant', preserve_range=True)
        shift = AffineTransform(translation=(0, shift))
        shifted = warp(scaled, shift, mode='constant', preserve_range=True)
        cval = 1 if target.dtype == np.int else 0
        shifted_target = warp(scaled_target, shift, mode='constant', preserve_range=True, cval=cval)
        sample['input'] = shifted.astype(image.dtype)
        sample['target'] = shifted_target.astype(target.dtype)
        return sample

class BaseNormalize():
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, x):
        return self.normalize(x)

class InputNormalize(BaseNormalize):
    def __call__(self, sample):
        x = sample['input']
        sample['input'] = super().__call__(x)
        return sample

class TargetNormalize(BaseNormalize):
    def __call__(self, sample):
        x = sample['target']
        sample['target'] = super().__call__(x)
        return sample

class Clamp():
    def __call__(self, sample):
        x = sample['input']
        sample['input'] = torch.clamp(sample['input'], 0.0, 1.0)
        sample['target'] = torch.clamp(sample['target'], 0.0, 1.0)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        return {'input': torch.from_numpy(input), 'target': torch.from_numpy(target)}

class FlipChannels(object):
    def __init__(self, only_input=False):
        self.only_input = only_input

    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        # swap channel axis
        input = input.transpose((2, 0, 1))
        if not self.only_input:
            target = target.transpose((2, 0, 1))
        return {'input': input, 'target': target}


class Resize():
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, sample):
        wx, wy = self.target_size
        wx0, wy0, _ = sample['input'].shape
        sample['input'] = transform.resize(sample['input'], self.target_size, preserve_range=True)
        sample['target'] = transform.resize(sample['target'], self.target_size, preserve_range=True)
        return sample

class MinResize(Resize):
    def __init__(self, min_size=256):
        self.min_size = min_size
        super().__init__()

    def __call__(self, sample):
        wx0, wy0, _ = sample['input'].shape
        min_dim = min(wx0, wy0)
        k = 1
        if min_dim < self.min_size:
            k = self.min_size / min_dim
        self.target_size = k * np.array((wx0, wy0))
        return super().__call__(sample)


class ChangeType():
    def __init__(self, problem='regr'):
        self.problem = problem
    def __call__(self, sample):
        sample['input'] = sample['input'].astype(np.float32)
        if self.problem == 'regr':
            sample['target'] = sample['target'].astype(np.float32)
        else:
            sample['target'] = sample['target'].astype(np.int)
        return sample

class Scale():
    def __init__(self, problem='regr'):
        self.problem = problem
    def __call__(self, sample):
        sample['input'] = sample['input'] / 255.
        if self.problem == 'regr':
            sample['target'] = sample['target'] / 255.
        return sample

class RandomCrop():
    def __init__(self, target_size=(224, 224), edge=5):
        self.target_size = target_size
        self.edge = edge

    def __call__(self, sample):
        #         if min(sample['target'].shape) < max(self.target_size) + 2*self.edge:
        #             sample = Resize(self.target_size)(sample)
        #             return sample
        wx, wy = self.target_size
        wx0, wy0, _ = sample['target'].shape
        try:
            center_x = np.random.randint(self.edge + wx // 2, wx0 - self.edge - wx // 2)
            center_y = np.random.randint(self.edge + wy // 2, wy0 - self.edge - wy // 2)
        except:
            raise ValueError('error', sample['target'].shape)
        crop_x_0 = center_x - wx // 2
        crop_x_1 = center_x + wx // 2
        crop_y_0 = center_y - wy // 2
        crop_y_1 = center_y + wy // 2
        sample['input'] = sample['input'][crop_x_0:crop_x_1, crop_y_0:crop_y_1]
        sample['target'] = sample['target'][crop_x_0:crop_x_1, crop_y_0:crop_y_1]
        return sample