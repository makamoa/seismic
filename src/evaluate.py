from metrics import ConfusionMatrix, RMSE
import torch, torchvision
import time
import datetime
from models.build import build_model
from noiseadding import build_noise_transforms, CombinedTransforms
from data import get_train_val_dataset, get_dataset, get_train_val_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
import argparse
from attacks import fgsm, pgd_linf
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

METADATA = '../metadata/'
EVALDATA = os.path.join(METADATA, 'evaluation/')

class Evaluator:
    def __init__(self, weight_file, device="cpu", activation=None, batch_size=8):
        self.weight_file = weight_file[:-4] if weight_file[:-4] == '.pkl' else weight_file
        self.batch_size = batch_size
        self.model_type, self.problem = weight_file.split('_')[:2]
        if self.problem == 'firstbreak':
            self.metrics = ConfusionMatrix(2, ["empty", "firstbreak"])
            self.loss_type = nn.CrossEntropyLoss
        else:
            self.metrics = RMSE()
            self.loss_type = nn.MSELoss
        self.model = build_model(self.model_type, self.problem, activation=activation)
        self.device = torch.device(device)
        self.model.to(device)
        self.save_path = os.path.join(METADATA, weight_file + '.pkl')
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        self.figdir = os.path.join(EVALDATA, 'figures/')

    def prepare_loader(self, noise_type=-1, noise_scale=0.25):
        noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
        denoise_dataset = get_dataset(self.problem, noise_transforms=noise_transforms)
        _, val_dataset = get_train_val_dataset(denoise_dataset, generator=torch.Generator().manual_seed(42), valid_split=0.01)
        loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def prepare_real_loader(self):
        dataset = get_dataset('real')
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        return loader

    def evaluate_model(self, eval_type='synthetic', noise_type=-1, noise_scale=0.25, **eval_args):
        if eval_type == 'synthetic':
            loader = self.prepare_loader(noise_type, noise_scale)
            eval = self.evaluate
        else:
            loader = self.prepare_real_loader()
            eval = self.eval_real
        return eval(loader, prefix=f'{noise_type}_{noise_scale}_', **eval_args)

    def evaluate(self, loader, plot=True, to_file=True, attack=None, prefix='', **att_args):
        self.metrics.reset()
        for i, (sample) in enumerate(loader):
            x, y = sample['input'].to(self.device), sample['target'].to(self.device)
            if attack is not None:
                delta = attack(self.model, x, y, loss_fn=self.loss_type, **att_args)
                x += delta
            else:
                delta = None
            with torch.no_grad():
                y_pred = self.model(x)
                if self.problem == 'firstbreak':
                    y_pred = torch.argmax(y_pred, dim=1)  # get the most likely prediction
            self.metrics.add_batch(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            print('_', end='')
        if plot:
            self.plot_predictions(x, y, y_pred, delta, epsilon=att_args.get('epsilon', 0.1), to_file=to_file, prefix=prefix)
        return self.metrics.get()

    def plot_noise_palette(self):
        fig, axes = plt.subplots(7,4, figsize=[10,17])
        for i, noise_type in enumerate(range(-1, 6)):
            for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
                loader = self.prepare_loader(noise_type, noise_scale)
                sample = iter(loader).__next__()
                x = sample['input'].to(self.device)
                axes[i,j].imshow(x[0,0], cmap='seismic')
                axes[i,j].axis('off')
        fname = os.path.join(self.figdir, 'noise_pallette.jpg')
        fig.savefig(fname)
        plt.close(fig)

    def evaluate_robustness(self, plot=True, to_file=True, only_linear=False, fast=False, **eval_args):
        self.robustness = np.ones([7, 4]) * -1
        ### how many noise types to use
        if only_linear:
            n=4
        elif fast:
            n=3
        else:
            n=6
        for i, noise_type in enumerate(range(-1, n)):
            for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
                if i == 0 and j != 0:
                    continue
                self.robustness[i, j] = self.evaluate_model(noise_type=noise_type, noise_scale=noise_scale, plot=plot, **eval_args)
                print(noise_type, noise_scale, self.robustness[i, j])
        attack = eval_args.get('attack',None)
        fname = os.path.join(EVALDATA, 'robustness_' + self.weight_file + '.npy') if attack is None else \
            os.path.join(EVALDATA, 'robustness_' + f"attacked{attack.epsilon}_" + self.weight_file + '.npy')
        np.save(fname, self.robustness)
        if plot:
            self.plot_robustness(attack=attack,to_file=to_file)

    def plot_robustness(self, from_file=False, filename=None, attack=None, to_file=True):
        if filename is None:
            filename = 'robustness_' + self.weight_file
        if from_file:
            robustness = np.load(os.path.join(EVALDATA, filename+'.npy'))
        else:
            robustness = self.robustness
        if robustness.shape[:1] == 4:
            indices = ['gauss+color', '+linear', '+fft', '+hyperbolic',]
        else:
            indices = ['clean', 'gauss+color', '+linear', '+fft', '+hyperbolic', '+bandpass', '+trace']
        df_cm = pd.DataFrame(robustness, index=[i for i in indices],
                             columns=[i for i in [1, 2, 4, 8]])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cbar=False, vmin=robustness[robustness>0].min())
        plt.xlabel('noise to signal')
        plt.ylabel('noise types')
        if to_file:
            fname = os.path.join(self.figdir, "robust/", filename + '.jpg') if attack is None else \
                os.path.join(self.figdir, "robust/", f"attacked{attack.epsilon}_" + filename + '.jpg')
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()

    def eval_real(self, loader, plot=True, to_file=True, **kwargs):
        self.metrics.reset()
        for i, (sample) in enumerate(loader):
            x, y = sample['input'].to(self.device), sample['target'].to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
                if self.problem == 'firstbreak':
                    y_pred = torch.argmax(y_pred, dim=1)  # get the most likely prediction
            print('_', end='')
        np.save(EVALDATA + 'real_x.npy', x.detach().cpu().numpy())
        np.save(EVALDATA + 'real_y.npy', y.detach().cpu().numpy())
        np.save(EVALDATA + 'real_y_pred.npy', y_pred.detach().cpu().numpy())
        if plot:
            self.plot_real(x, y, y_pred, to_file=to_file)
        return None

    def plot_real(self, *pargs, **kwargs):
        self.plot_predictions(*pargs, **kwargs, prefix='real_')

    def plot_predictions(self, x, y, y_pred, delta=None, epsilon=0.1, to_file=True, prefix=''):
        num = 3 if delta is None else 4
        fig, axes = plt.subplots(num, 1, figsize=[9*num, 18])
        fig.suptitle('model=' + self.weight_file)
        axes[0].imshow(torchvision.utils.make_grid(x, padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic')
        axes[0].set_title("input")
        if self.problem == 'firstbreak':
            y = y.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
        axes[1].imshow(
            torchvision.utils.make_grid(y.float(), padding=0)[0][None, ...].permute((1, 2, 0)),
            cmap='seismic')
        axes[1].set_title("target")
        axes[2].imshow(torchvision.utils.make_grid(y_pred.float(), padding=0)[0][None, ...].permute((1, 2, 0)),
                   cmap='seismic')
        axes[2].set_title("prediction")
        if delta is not None:
            axes[3].imshow(torchvision.utils.make_grid(delta.detach(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
            axes[3].set_title("attack")
        if to_file:
            fname = os.path.join(self.figdir, "preds/", prefix + self.weight_file + '.jpg') if delta is None else \
                os.path.join(self.figdir, "preds/", prefix + f"attacked{epsilon}_" + self.weight_file + '.jpg')
            fig.savefig(fname)
            plt.close(fig)
        else:
            fig.show()

    def save_robustness(self):
        pass

def evaluate_models(model_type, problem, attack_type='none', **eval_args):
    for i, noise_type in enumerate(range(-1, 6)):
        for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
            fname = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_attack_{attack_type}_pretrained_False'
            if not os.path.exists(os.path.join(METADATA, fname + '.pkl')):
                continue
            else:
                "Model not found!"
                print(fname)
            print("Model was found!")
            print(fname)
            eval = Evaluator(fname)
            eval.evaluate_robustness(**eval_args)
            print("Succeed!")
            print(fname)

if __name__ == "__main__":
    #eval.plot_robustness(from_file=True)
    for model_type in ['swin', 'unet']:
        for problem in ['denoise', 'firstbreak']:
            for attack in ['none', 'fgsm']:
                for eval_attack in [None, fgsm]:
                    evaluate_models(model_type, problem, attack_type=attack, attack=eval_attack)
    #eval.plot_noise_palette()
    # print(eval.model_type, eval.problem)
    # eval.evaluate_model(eval_type='real')
    # for eval_attack in [None, fgsm]:
    #     eval.evaluate_robustness(plot=True, attack=eval_attack)
    #eval.evaluate_model(plot=True)