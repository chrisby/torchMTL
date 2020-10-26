""" Module with classes that wrap commonly used utility functions from pytorch
"""

import torch
from torch import nn

class Concat(nn.Module):
    """ Simple layer that wraps torch.cat
    
    Attributes
    ----------
    dim : int
        Which dimension to concatenate
    """

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *input):
        return torch.cat(input, dim=self.dim)
