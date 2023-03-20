###for ease if later want to add acq
import numpy as np
import math
from abc import ABC, abstractmethod
import torch

#TO DO: expand acqusition functions

def gen_acquisition(gp, acq_type = 'EI', maximize = True):
    r"""helper function to initiate a acquisition function class.
    """
    if acq_type == 'EI':
        return ExpectedImprovement(gp_model = gp, maximize = maximize)
    elif acq_type == 'SPE':
        return SquaredPosteriorError(gp_model = gp)
    else:
        raise ValueError('not a valid acquisition function')


class AcquisitionFunction(ABC):
    r"""Abstract base class for bayesian optimization acquisition functions.
    """

    def __init__(self, gp_model, maximize = True):
        r"""Constructor for the base class of acquisition functions.

        Args:
            gp_model (class): A Gaussian Process model
            maximize (bool): whether the goal is maximizing or minimizing
        """
        super().__init__()
        self.gp_model = gp_model
        self.maximize = maximize

    @abstractmethod
    def forward(self, candidate):
        r"""Evaluate the acquisition function on the candidate
        """
        pass  

    def opt_elements(self, seq):
        r"""A helper function to return list of position(s) of smallest element 
        Args:
            seq (nx1array): values that should be compared to find the answer
        Return:
            index of all instances of the array's maximum or minimum value
        """
        indices = []
        if not self.maximize:
            seq = -seq
        opt_val = seq[0]
        for i in range(0, len(seq)):
            if seq[i] == opt_val:
                indices.append(i)
            elif seq[i] > opt_val:
                opt_val = seq[i]
                indices = [i]
        return indices


class ExpectedImprovement(AcquisitionFunction):
    r"""Expected Improvement acquisition function (Mockus) 

	Computes Expected Improvement over the current best observed value
    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
         >>>  gp_model = custom_gp.GaussianProcess(train_X, train_Y)
         >>>  EI = acquisition.ExpectedImprovement(gp_model, best_f = 0.2)
         >>>  output = EI(test_X)
    """
    def __init__(self, gp_model, maximize = True):
        r"""Expected Improvement with observation noise

        Args:
            gp_model (class): A Gaussian Process model
            maximize (boolean): If True, consider the problem a maximization problem
        """
        super().__init__(gp_model = gp_model, maximize = maximize)
        if self.maximize:
            self.best_f = torch.min(gp_model.train_targets)
        else:
            self.best_f = torch.max(gp_model.train_targets)

    def forward(self, candidate):
        r"""Evaluate the acquisition function on the candidate.
        """
        candidate = torch.as_tensor(candidate, device = self.gp_model.device) #not in GP subspace
        posterior = self.gp_model(candidate) #converts not into GP subspace
        mean = posterior.mean #in GP subspace
        sigma = posterior.variance.clamp_min(1e-9).sqrt() #in GP subspace
        u = (mean - self.best_f) / sigma #GP subspace - GP subspace / GP_subspace
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        acq_values = sigma * (updf + u * ucdf)
        if len(acq_values.shape) == 2:
            acq_values = acq_values[0,:]
        return acq_values, self.opt_elements(acq_values)


class SquaredPosteriorError(AcquisitionFunction):
    r"""Acquisition function based on the predictive variance

    Example:
         >  >  >  gp_model = SingleTaskGP(train_X, train_Y)
         >  >  >  SPE = SquaredPosteriorError(gp_model)
         >  >  >  value = SPE(test_X)
    """

    def __init__(self, gp_model):
        r"""Squared Posterior Error

        Args:
            gp_model (class): A Gaussian Process model
        """
        super().__init__(gp_model = gp_model)

    def forward(self, candidate):
        r"""Evaluate the acquisition function on the candidate.

        Args:
            candidate (Tensor): a set of design points to evaluate

        Returns:
            A Tensor of acquisition function values at the given design points
        """
        
        posterior = self.gp_model.posterior(candidate)
        acq_values = posterior.variance.clamp_min(1e-9)
        return acq_values, self.opt_elements(acq_values)
