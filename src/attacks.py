import torch
from torch import nn

def fgsm(model, X, y, loss_fn=nn.CrossEntropyLoss, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    fgsm.epsilon = epsilon
    loss = loss_fn()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

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