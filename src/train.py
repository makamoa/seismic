from metrics import ConfusionMatrix, RMSE
import torch, torchvision
import time
import datetime
from models.build import build_model, model_load_weights
from noiseadding import build_noise_transforms, CombinedTransforms
from data import get_train_val_dataset, get_dataset, get_train_val_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
import argparse
from attacks import fgsm, pgd_linf
import yaml

METADATA = '../metadata/'
with open(os.path.join("../config/", "global_config.yml"), 'r') as stream:
    data_loaded = yaml.safe_load(stream)
SEISMICROOT = data_loaded['SEISMICROOT']
LOGSROOT = data_loaded['LOGSROOT']

# Used to keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_(model, problem, loss_type, metrics,
           train_loader, valid_loader, device,
          epochs=30, learning_rate=5e-5,
          reports_per_epoch = 10, tb=None, attack=None, att_args={}):
    iter_per_epoch = len(train_loader)
    iter_per_report = iter_per_epoch // reports_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_type()
    ### save input to tensorboard
    if tb is not None:
        sample = iter(valid_loader).next()
        input_ = torchvision.utils.make_grid(sample['input'], padding=1)[0][None, ...]
        if problem == 'denoise':
            target_ = torchvision.utils.make_grid(sample['target'].float(), padding=1)[0][None, ...]
        else:
            target_ = torchvision.utils.make_grid(sample['target'].float().unsqueeze(1), padding=1)[0][None, ...]
        tb.add_image("inputs", input_)
        tb.add_image("targets", target_)

    for epoch in range(epochs):
        model.train()
        # Progress reporting
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        end = time.time()

        for i, (sample) in enumerate(train_loader):

            # Load a batch and send it to GPU
            x = sample['input'].to(device)
            y = sample['target'].to(device)
            # Forward pass: compute predicted y by passing x to the model.
            # add adversarial attacks
            if attack is not None:
                x += attack(model, x, y, loss_type, **att_args)
            y_pred = model(x)
            # Compute and print loss.
            loss = loss_fn(y_pred, y)

            # Record loss
            losses.update(loss.data.item(), x.size(0))

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model).
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            # Log training progress
            if i % iter_per_report == 0:
                print('\nEpoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t' 'ETA {eta}\t'
                 'Training Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
            elif i % (iter_per_report) == 0:
                print('.', end='')

            #break # useful for quick debugging
        torch.cuda.empty_cache(); del x, y;
        # Validation after each epoch
        model.eval()
        metrics.reset()
        for i, (sample) in enumerate(valid_loader):
            x, y = sample['input'].float().to(device), sample['target'].to(device)
            if attack is not None:
                delta = attack(model, x, y, loss_fn=loss_type, **att_args)
                x += delta
            with torch.no_grad():
                y_pred = model(x)
                if problem == 'firstbreak':
                    y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction
            metrics.add_batch(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            print('_', end='')
        print(f'\nValidation stats ({metrics.name}):', metrics.get())
        if tb is not None:
            tb.add_scalar("Loss", losses.avg, epoch)
            tb.add_scalar(f"{metrics.name}", metrics.get(), epoch)
            if problem == 'denoise':
                input_ = torchvision.utils.make_grid(x, padding=1)[0][None, ...]
                preds_ = torchvision.utils.make_grid(y_pred, padding=1)[0][None, ...]
            else:
                input_ = torchvision.utils.make_grid(x, padding=1)[0][None, ...]
                preds_ = torchvision.utils.make_grid(y_pred.unsqueeze(1), padding=1)
            tb.add_image("inputs", input_, global_step=epoch)
            tb.add_image("preds", preds_, global_step=epoch)
            if attack is not None:
                attack_ = torchvision.utils.make_grid(delta, padding=1)[0][None, ...]
                tb.add_image("attack", attack_, global_step=epoch)


def train_denoise(model_type='unet', noise_type=-1, noise_scale=0, gpu_id=0,
                  epochs=30, learning_rate=5e-5, batch_size=8, workers=4, attack=None, pretrained=None, dataclip=True, prefix='', rootdir=None, **train_args):
    model = build_model(model_type, 'denoise')
    is_pretrained = True if pretrained else False
    if is_pretrained:
        load_path = os.path.join(METADATA, pretrained)
        model , state = model_load_weights(model, load_path=load_path)
        if not state:
            is_pretrained=False
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)
    noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
    denoise_dataset = get_dataset('denoise', noise_transforms=noise_transforms, rootdir=rootdir)
    train_dataset, val_dataset = get_train_val_dataset(denoise_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    attack_type = 'none' if attack is None else attack.__name__
    run_id = f'{prefix}{model_type}_denoise_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_{dataclip}_attack_{attack_type}_pretrained_{is_pretrained}'
    save_path = os.path.join(METADATA, run_id + '.pkl')
    tb = SummaryWriter(os.path.join(LOGSROOT, 'denoise/', run_id))
    loss_fn = nn.MSELoss
    metrics = RMSE()
    train_(model, 'denoise', loss_fn, metrics,
           train_loader, valid_loader, device,
           epochs=epochs, learning_rate=learning_rate, tb=tb, attack=attack, **train_args)
    torch.save(model.state_dict(), save_path)
    tb.close()
    print('\nTraining done. Model saved ({}).'.format(save_path))

def train_first_break(model_type='unet', noise_type=-1, noise_scale=0, gpu_id=0,
                  epochs=10, learning_rate=5e-5, batch_size=8, workers=4, attack=None, pretrained=None, dataclip=True, prefix='', rootdir=None, **train_args):
    model = build_model(model_type, 'firstbreak')
    is_pretrained = True if pretrained else False
    if is_pretrained:
        load_path = os.path.join(METADATA, pretrained)
        model , state = model_load_weights(model, load_path=load_path)
        if not state:
            is_pretrained=False
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)
    noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
    denoise_dataset = get_dataset('firstbreak', noise_transforms=noise_transforms, rootdir=rootdir)
    train_dataset, val_dataset = get_train_val_dataset(denoise_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    attack_type = 'none' if attack is None else attack.__name__
    run_id = f'{prefix}{model_type}_firstbreak_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_{dataclip}_attack_{attack_type}_pretrained_{is_pretrained}'
    save_path = os.path.join(METADATA, run_id + '.pkl')
    tb = SummaryWriter(os.path.join(LOGSROOT, 'firstbreak/', run_id))
    loss_fn = nn.CrossEntropyLoss
    metrics = ConfusionMatrix(2, train_loader.dataset.dataset.class_names)
    train_(model, 'firstbreak', loss_fn, metrics,
           train_loader, valid_loader, device,
           epochs=epochs, learning_rate=learning_rate, tb=tb, attack=attack, **train_args)
    torch.save(model.state_dict(), save_path)
    tb.close()
    print('\nTraining done. Model saved ({}).'.format(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--noise_type", type=int, default=0)
    parser.add_argument("--noise_scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--dataclip", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    if args.dataclip:
        rootdir = os.path.join(SEISMICROOT, 'data/')
    else:
        rootdir = os.path.join(SEISMICROOT, 'normalized/')
    att_args = {}
    if args.attack is None:
        attack = None
    elif args.attack == 'fgsm':
        attack = fgsm
        att_args['epsilon'] = args.epsilon
        if args.noise_type != -1:
            # scale attack noise to be the same signal-to-noise
            att_args['epsilon'] *= min(1 / (args.noise_scale+1e-12) / 4, 1)
    else:
        attack = pgd_linf
        att_args['epsilon'] = args.epsilon
        att_args['alpha'] = args.alpha
        if args.noise_type != -1:
            # scale attack noise to be the same signal-to-noise
            att_args['epsilon'] *= min(1 / (args.noise_scale+1e-12) / 4, 1)
    if args.problem == 'denoise':
        train_denoise(args.model, args.noise_type, args.noise_scale,args.device, epochs=args.epochs, attack=attack, pretrained=args.pretrained, att_args=att_args, dataclip=args.dataclip, prefix=args.prefix, batch_size=args.batch_size, rootdir=rootdir)
    else:
        train_first_break(args.model, args.noise_type, args.noise_scale, args.device, epochs=args.epochs, attack=attack, pretrained=args.pretrained, att_args=att_args, dataclip=args.dataclip, prefix=args.prefix, batch_size=args.batch_size, rootdir=rootdir)