import torch
from torch import nn
from skimage.draw import random_shapes
import numpy as np

def fgsm(model, X, y, loss_fn=nn.CrossEntropyLoss, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    fgsm.epsilon = epsilon
    loss = loss_fn()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def fgsm_domain_fft(model, X, y, epsilon=0.1, max_shapes=5, **rand_args):
    """ Construct in-domain FGSM adversarial examples on the examples X"""
    inp =  torch.rand_like(X, requires_grad=True)
    delta = torch.fft.fft2(inp)
    delta = torch.fft.fftshift(delta)
    bs, ch, wx, wy = delta.shape
    image, _ = random_shapes((wx, wy), shape='rectangle', max_shapes=max_shapes, channel_axis=None, **rand_args)
    image[image<255] = 1
    image[image==255] = 0
    image = np.repeat(image[None, None,...],bs,0)
    mask = torch.from_numpy(image).to(X.device)
    delta = torch.mul(delta, mask)
    delta = torch.fft.ifft2(torch.fft.ifftshift(delta))
    delta = torch.real(delta)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * inp.grad.detach().sign()

def pgd_linf(model, X, y, loss_fn=nn.CrossEntropyLoss, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = loss_fn()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha *delta.grad.detach().sign()).clamp(-epsilon ,epsilon)
        delta.grad.zero_()
    return delta.detach()