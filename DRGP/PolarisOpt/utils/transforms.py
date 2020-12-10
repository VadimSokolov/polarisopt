import os, sys
import numpy as np
import torch

def normalize(sample, s_range):
    r"""takes a sample and returns the normalized version for ease
    """
    if torch.is_tensor(sample):
        s_range = torch.as_tensor(s_range)
        return torch.div((sample - s_range[:, 0]), (s_range[:, 1] - s_range[:, 0]))
    else:
        return np.divide((sample - s_range[:, 0]), (s_range[:, 1] - s_range[:, 0]))

def inverse_normalize(scaled_sample, s_range):
    r"""takes a scaled original input and returns the normalized version for ease
    """
    if torch.is_tensor(scaled_sample):
        s_range = torch.as_tensor(s_range)
    return scaled_sample*(s_range[:, 1] - s_range[:, 0]) + s_range[:, 0]

def standardize(sample, s_mean, s_std):    
    r"""takes a sample and returns the standardized version for ease
    """
    if torch.is_tensor(sample):
        s_mean, s_std = torch.as_tensor(s_mean), torch.as_tensor(s_std)
        return torch.div(sample - s_mean, s_std)
    else:
        return np.divide(sample - s_mean, s_std)

def inverse_standardize(sample, s_mean, s_std):
    r"""takes a scaled original input and returns the standardized version for ease
    """
    if torch.is_tensor(sample):
        s_mean, s_std = torch.as_tensor(s_mean), torch.as_tensor(s_std)
    return (sample*s_std) + s_mean

def calc_stats(sample):
    r"""takes a sampleset and returns its mean and standard deviation for ease
    """
    if torch.is_tensor(sample):
        std = sample.std(dim = 0)
        mean = sample.mean(dim = 0)
    else:
        std = sample.std(axis = 0)
        mean = sample.mean(axis = 0)
    std[std == 0] = 1
    return mean, std
