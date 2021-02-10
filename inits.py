import numpy as np
import torch
import torch.nn as nn

def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = nn.Parameter(torch.nn.init.uniform_(tensor=torch.empty(shape), a=-init_range, b=init_range))
    return initial


def kaiming(shape): # not used
    initial = nn.Parameter(torch.nn.init.kaiming_uniform_(tensor=torch.empty(shape), mode='fan_in', nonlinearity='relu'))
    return initial


def xavier(shape):
    initial = nn.Parameter(torch.nn.init.xavier_uniform_(tensor=torch.empty(shape), gain=1))
    return initial

