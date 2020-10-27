""" Module with classes that wrap commonly used utility functions from pytorch
"""

import torch
from torch import nn

class Concat:
    """ Simple class that wraps torch.cat
    
    Attributes
    ----------
    dim : int
        Which dimension to concatenate
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, *input):
        return torch.cat(input, dim=self.dim)

class SimpleSelect:
    """ Simple selection class that wraps the selection of the first
    dimension
    
    Attributes
    ----------
    sel : int
        Which element to select
    """

    def __init__(self, selection_axis):
        self.sel = selection_axis
        
    def __call__(self, X):
        return X[self.sel]

class SliceSelect:
    """ Selection class that wraps the indexing of Sequentials via slices
    
    Attributes
    ----------
    slice : slice
        Slice to index input
    """

    def __init__(self, sel_slice):
        self.slice = sel_slice
        
    def __call__(self, X):
        return X[self.slice]
