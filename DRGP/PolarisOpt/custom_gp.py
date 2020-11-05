#! /usr/bin/env python3

r"""
Gaussian Process Regression models based on GPyTorch models.
"""
import os
import torch
import dill
import math
import numpy as np

from gpytorch.means import Mean
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.torch_priors import GammaPrior

from . import nn as NN
from .utils import transforms
from .utils.objective_funcs import run_objective

torch.set_default_tensor_type(torch.DoubleTensor)

class GaussianProcess(ExactGP):
    r"""A GP model leveraging GPytorch.
    """

    def __init__(self, train_X, train_Y, likelihood = None, covar_module = None, mean_module = None, pref_cpu = False):
        r"""GP model container class

        Args:
            train_X: 'n x d' tensor or numpy array of training inputs
            train_Y: 'n x 1' tensor or numpy array of training outputs
            likelihood: A likelihood. If omitted, uses a standard GaussianLikelihood with inferred noise level.
            covar_module: The covariance (kernel) matrix. If omitted, uses a MaternKernel.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim = 1, keepdim = True)
            >>> model = GaussianProcess(train_X, train_Y)
        """
        
        if torch.cuda.is_available() and not pref_cpu:
            self.device = 'cuda'
            print("Running on the GPU")
        else:
            self.device = 'cpu'
            print("Running on the CPU")

        train_X = torch.as_tensor(train_X, device = self.device)
        train_Y = torch.as_tensor(train_Y, device = self.device)
        
        self.x_mean, self.x_std = transforms.calc_stats(train_X)
        self.y_mean, self.y_std = transforms.calc_stats(train_Y)
        #TO DO: Update this and GP to have an 'int', 'float' designator list depending on GP transformations
        if likelihood is None:
            likelihood = GaussianLikelihood(
                noise_prior = GammaPrior(1.1, 0.05)
                )
        ExactGP.__init__(self, transforms.standardize(train_X, self.x_mean, self.x_std), transforms.standardize(train_Y, self.y_mean, self.y_std)[:, 0], likelihood)
        if mean_module is None:
            self.mean_module = ConstantMean()
        else:
            self.mean_module = mean_module

        if hasattr(mean_module, "exclude_hype"):
            self.exclude_mean = mean_module.exclude_hype
        else:
            self.exclude_mean = False

        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu = 2.5, 
                    ard_num_dims = train_X.shape[-1], 
                    lengthscale_prior = GammaPrior(3.0, 6.0), 
                ), 
                outputscale_prior = GammaPrior(2.0, 0.15), 
            )
        else:
            self.covar_module = covar_module

    def forward(self, x):
        x_0 = transforms.standardize(x, self.x_mean, self.x_std)
        
        if isinstance(self.mean_module, Mean_NN):
            mean_x = self.mean_module(x) ##currently only way know to give it... ask Randy
            mean_x = transforms.standardize(torch.as_tensor(mean_x, device = self.device), self.y_mean, self.y_std)
        else:
            mean_x = self.mean_module(x_0) 
        
        covar_x = self.covar_module(x_0)
        return MultivariateNormal(mean_x, covar_x)

    def fantasize(self, inputs):
        inputs = torch.as_tensor(inputs, device = self.device) #in DR range
        posterior = self.forward(inputs) #results in GP range
        fake_y = self.likelihood(posterior).loc.detach()[:, None] #in GP range
        stdize_x = transforms.standardize(inputs, self.x_mean, self.x_std)  #must convert inputs to GP range
        return self.get_fantasy_model(stdize_x, fake_y)

        

class Mean_NN(Mean):
    r"""Deep Neural Network constructed mean

    This computes a deterministic approximation of the input-output relationship in the original subspace
    to provide an initial mean value for the GP
    
    Example:
        >>> mean_model = Mean_NN(NN_mean_var, DR_model)
        >>> CustomGP(train_X, train_Y, mean_module = mean_model)
    """
    def __init__(self, NN_mean_var, DR_model):
        r"""Args:
        NN_mean_var: the necessary variables to construct the NN: 
            dim_in: number of input dimensions
            dim_out: number of output dimensions (can be >1)
            epochs_m: number of loops to do for training
            learning_rate_m: learning rate for training of NN on training set
            layers_m: the number of nodes per layer between the inputs and outputs, ex [10, 100, 10]
        DR_model: The dimension reduction model if there is one
        """
        super(Mean_NN, self).__init__()

        # if torch.cuda.is_available():
        #     self.device = 'cuda'
        # else:
        #     self.device = 'cpu'
        self.NN_func = NN.FCN_Network(*NN_mean_var[:2], *NN_mean_var[3:])#.to(self.device)
        self.exclude_hype = True
        self.DR_func = DR_model
        self.NN_mean_var = NN_mean_var
        self.orig_range = None
        self.obj_type = None

    def calculate(self, problem_info, pr = False):
        self.obj_type = problem_info.obj_type
        train_X, train_Y = problem_info.load_results_orig #translated subspace data
        self.orig_range = problem_info.orig_range[0]
        X_0 = transforms.normalize(train_X, self.orig_range)
        self.norm_range = np.c_[np.zeros(problem_info.dim_in), np.ones(problem_info.dim_in)]

        #####################################################
        #Step 2: train NN model                             #
        #####################################################
        NN.train_NN(X_0, train_Y, self.NN_mean_var[2], self.NN_func, pr)

        return print('Mean_NN model created')

    def forward(self, input_set):
        #receive it from the GP unstandardized 
        #####################################################
        #Step 1: convert from DR to orig subspace  & norm   #
        #####################################################
        device=input_set.device
        input_set = self.DR_func.decode_X(input_set.cpu().numpy())
        X_0 = transforms.normalize(input_set, self.orig_range)
        #####################################################
        #Step 2: predict y                                  #
        #####################################################
        # run through NN model to predict y in original y subspace
        ys = self.NN_func.forward(torch.as_tensor(X_0,device=device))
        #####################################################
        #Step 3: derive GP objective                        #
        #####################################################
        y_obj, _ = run_objective(ys.cpu().detach().numpy(),self.obj_type)

        return torch.as_tensor(y_obj)[:,0]

    def tune(self, problem_info, pr= False):
        #fake holder for now
        return self.calculate(problem_info, pr)

     

     











# #cheat function
# def Predict_CI(train_x, stats_X, train_y, stats_Y, test_x, M_model = None):
#     r"""A helper function to return the predicted values + confidence intervals on demand

#     Args:
#         train_x (ndarray or tensor): normalized training values to initialize the GP over (subspace inputs if applicable)
#         stats_X (list): a list containing the mean and std of the input training values used to normalize data
#         train_y (ndarray or tensor): normalized training output values to initialize the GP over
#         stats_y (list): a list containing the mean and std of the output training values used to normalize data
#         test_x (ndarray or tensor): normalized testing values to predict for
#         M_model (class): a class container holding the educated mean function

#     Return:
#         predicted mean and confidence interval for testing values test_x
#     """
#     #ensure train_x and train_y and test_x are all standardized for best results
#     train_x = torch.as_tensor(train_x)
#     train_y = torch.as_tensor(train_y)
#     test_x = torch.as_tensor(test_x)

#     gp = initialize_GP(train_x, stats_X, train_y, stats_Y, M_model)
#     with torch.no_grad():
#         # compute posterior
#         posterior = gp.posterior(test_x, observation_noise = True)
#         #mean
#         mu = posterior.mean.detach().numpy()
#         # covar matrix
#         lower, upper = posterior.mvn.confidence_region()
#     return mu, lower.detach().numpy(), upper.detach().numpy()

