"""
    This file contains the master calls to implement mini-BO calibration of the NN portions:
        * Calibrate_DRNN - provides recommended network structures for the Dim-Red NN subspace
        * Calibrate_Mean_NN - provides recommended network structures for the GP NN mean
"""

import os, sys
from abc import ABC, abstractmethod
import numpy as np
import threading, shutil
import time, torch
from PolarisOpt import custom_gp as cgp
from . import sampler
from . import archiver
from . import util
from . import transforms
from PolarisOpt import eval_sim
from PolarisOpt import bo
from PolarisOpt import dim_red
import copy

class Calibrate_NN(ABC):
    r"""Abstract base class for running a bayesian optimization script for finding
        the best NN architecture for different types of nn
        """
    def __init__(self, BO_var, fixed_var, res_filename):
        r"""Constructor for the base class of NNs

        Args:
    Args:
        BO_var: a nested list of each variable that is to be optimized in the syntax [variable_name, min_value, max_value, variable_type]
                possible values for variables currently accepted are based on specific class
        fixed_var: a nested list of each variable that should be given a specific value not equal to the defaults above
        res_filename: the name of the file to save to
        """
        super().__init__()
        self.BO_var = BO_var
        self.lr = .01
        self.epoch = 100
        self.seed = 0   
        self.set_attribute(fixed_var)
        self.res_filename = res_filename
        self._res_filepath = None
    
    def set_attribute(self, sets):
            for key, value in sets:
                if hasattr(self, key):
                    self.__dict__[key] = value
                else:
                    raise ValueError("Attribute %s is not a valid designator" % key)

    @abstractmethod
    def _translate_vars(self, variables):
        r"""returns a NN architecture based on variables and ranges
        """
        pass

    @abstractmethod
    def run(self):
        r"""runs the optimiation
        """
        pass

    def set_base(self, manager):
        pinfo = copy.deepcopy(manager)
        pinfo.update_parameter('res_filename', self.res_filename)
        self._res_filepath=pinfo._res_filepath
        l = [i[0] for i in self.BO_var]
        _, b = np.unique(l, return_index = True)
        min_range = [self.BO_var[i][1] if i in b else 0 for i in range(0, len(self.BO_var))]

        max_range = [i[2] for i in self.BO_var]
        pinfo.update_parameter('orig_range', [np.c_[min_range, max_range], [i[3] for i in self.BO_var]])

        #step 1: create an initial sample for BO if no training
        if not os.path.exists(self._res_filepath):
            variables = np.vstack(
                (
                    min_range, 
                    max_range, 
                    sampler.LHS_pool(pinfo.orig_range[0], 10, x_type = pinfo.orig_range[1])
                    )
                )
            temp = ''.join(["P " + ' '.join(map(str, v)) + "\n" for v in variables])

            with open(self._res_filepath, 'a+') as outfile:
                outfile.write(temp)
        return pinfo

    @abstractmethod
    def eval_loss(self, manager, variables, quiet = True):
        r"""returns the loss function over the learned dataset
        """
        pass

    def best_solution(self, wiggle_room = .05, y_range = None):
        NN_runs, _ = archiver.import_dataset(self._res_filepath, x_key = "orig_input", y_key = "objective")
        if y_range is None:
            return print(*[
                [i[0], *self._translate_vars(i[1:])] for i in NN_runs[
                NN_runs[:, 0]<= (NN_runs[:, 0].min() + (NN_runs[:, 0].min()*wiggle_room))]
                ]
                , sep = '\n')
        else:
            return print(*[
                [i[0], *self._translate_vars(i[1:])] for i in NN_runs[
                (NN_runs[:, 0]>= y_range[0]) &(NN_runs[:, 0]<= y_range[1])]
                ]
                , sep = '\n')



class Calibrate_DRNN(Calibrate_NN):
    r"""Performs a mini bayesian optimization to recommend the best DRNN structures to 

    Args:
        BO_var: a nested list of each variable that is to be optimized in the syntax [variable_name, min_value, max_value, variable_type]
                possible values for variables currently accepted are: 
                     - 'XDR_layer': an individual hidden layer between the input and reduced dimension layer; minimum value should be equal to the minimum number
                               of nodes that would be acceptable if the layer is chosen to be used (default = [])
                     - 'DRX_layer': an individual hidden layer between the reduced dimension layer and estimated reconstruction of inputs; minimum value should be equal to the minimum number
                               of nodes that would be acceptable if the layer is chosen to be used (default = [])
                     - 'DRY_layer': an individual hidden layer between the reduced dimension layer and estimated outputs; minimum value should be equal to the minimum number
                               of nodes that would be acceptable if the layer is chosen to be used (default = [])
                     - 'epoch': the number of epochs for the training of the network (default = 300)
                     - 'lr': learning rate for the training of the network (default = .001)
                     - 'dim_DR': the number of dimensions to reduce to (default = 2)
                     - 'seed': seed value that should be used to ensure consistancy across runs (default = 0)
        fixed_var: a nested list of each variable that should be given a specific value not equal to the defaults above

    Examples::

        >>> BO_var = [
            ['XDR_layer', 10, 200, 'int'], 
            ['XDR_layer', 10, 200, 'int'], 
            ['DRX_layer', 10, 200, 'int'], 
            ['DRY_layer', 10, 200, 'int'], 
            ['epoch', 100, 500, 'int'], 
            ['lr', .001, .01, 'float']
            ]
        >>> fixed_var = [
            ['dim_GP', 5], 
            ['seed', 0]
            ]
        >>> c = util.Calibrate_DRNN(BO_var, fixed_var)
        >>> c.run(manager, num_trials = 20, num_rec_points = 3)
    """

    def __init__(self, BO_var, fixed_var, res_filename):
        self.XDR_layer = []
        self.DRX_layer = []
        self.DRY_layer = []
        self.dim_DR = 1
        super().__init__(BO_var, fixed_var, res_filename)

    def _translate_vars(self, variables):
        XDR_layer = self.XDR_layer.copy()
        DRX_layer = self.DRX_layer.copy()
        DRY_layer = self.DRY_layer.copy()
        epoch = self.epoch
        dim_DR = self.dim_DR
        lr = self.lr
        seed = self.seed
        for x, r in zip(variables, self.BO_var):
            if (r[0] == 'XDR_layer') & (x>= int(r[1])):
                XDR_layer.append(int(round(x)))
            if (r[0] == 'DRX_layer') & (x>= int(r[1])):
                DRX_layer.append(int(round(x)))
            if (r[0] == 'DRY_layer') & (x>= int(r[1])):
                DRY_layer.append(int(round(x)))
            if (r[0] == 'lr'):
                lr = float(np.clip(x, r[1], r[2]))
            if (r[0] == 'dim_DR'):
                dim_DR = int(np.clip(round(x), r[1], r[2]))
            if (r[0] == 'epoch'):
                epoch = int(np.clip(round(x), r[1], r[2]))
            if (r[0] == 'seed'):
                seed = float(np.clip(x, r[1], r[2]))
        return [XDR_layer, DRX_layer, DRY_layer], lr, dim_DR, epoch, seed


    def run(self, manager, num_grid_points = 2000, num_trials = 30, num_rec_points = 3, acq_type = 'EI', quiet = True):
        pinfo = self.set_base(manager)
        pinfo.update_parameter('num_grid_points', num_grid_points)
        pinfo.update_parameter('num_rec_points', num_rec_points)
        pinfo.update_parameter('acq_type', acq_type)

        #step 2: use BO on NN_calb.txt over NN variables
        for l in range(0, num_trials+1):
            #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
            _, pend_samples = archiver.import_dataset(self._res_filepath)
            util.thread_it(self.eval_loss, [(manager, row, quiet) for row in pend_samples])
            if l<num_trials:
            #If less then the number of trials we run, run another Bayes set
                print("running loop number %d of %d" % (l+1, num_trials))
                bo.main_loop(pinfo)
        return self.best_solution()

    def eval_loss(self, manager, variables, quiet = True):
        layers, lr, dim_DR, epochs, seed = self._translate_vars(variables)
        count = np.sum([min(0, len(i)-1) for i in layers])
        _, _, _, NN_v, _ = archiver.load_DR_settings(manager._settings_filepath)
            #[epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
        NN_var = [
            manager.dim_in, 
            manager.dim_out, 
            epochs, 
            lr, 
            *NN_v[2:4], 
            *layers
            ]
        torch.random.manual_seed(seed)
        tDR_model = dim_red.NN_method(dim_DR, manager.orig_range, NN_var) #orig range = not the Nn inputs
        #NN_var = [dim_in, dim_out, epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
        train_X, train_Y = manager.load_training()
        tDR_model.calculate(train_X, train_Y, quiet = quiet)
        X_0 = tDR_model.prep_X(train_X)
        inputs, labels = torch.as_tensor(X_0), torch.as_tensor(train_Y)
        _, y_hat, x_hat = tDR_model.Model(inputs)
        loss = tDR_model.Model.error_function(
                x_hat.detach(), y_hat.detach(), 
                inputs, labels, 
                tDR_model.norm_range[:, 0], 
                tDR_model.norm_range[:, 1]
                )
        #save these in the original results file we keep
        y = loss.item() + (max(1, loss.item*.1)*count)
        if not quiet:
            print(y)
        archiver.update_record([variables], ["objective"], [[y]], self._res_filepath)


class Calibrate_Mean_NN(Calibrate_NN):
    r"""Performs a mini bayesian optimization to recommend the best GP NN mean structures to 

    Args:
        BO_var: a nested list of each variable that is to be optimized in the syntax [variable_name, min_value, max_value, variable_type]
                possible values for variables currently accepted are: 
                     - 'layer': an individual hidden layer between the input and estimated outputs; minimum value should be equal to the minimum number
                               of nodes that would be acceptable if the layer is chosen to be used (default = [])
                     - 'epoch': the number of epochs for the training of the network (default = 300)
                     - 'lr': learning rate for the training of the network (default = .001)
                    - 'seed': seed value that should be used to ensure consistancy across runs (default = 0)
        fixed_var: a nested list of each variable that should be given a specific value not equal to the defaults above

    Examples::

        >>> BO_var = [
            ['layer', 10, 200, 'int'], 
            ['layer', 10, 200, 'int'], 
            ['layer', 10, 200, 'int'], 
            ['layer', 10, 200, 'int'], 
            ['epoch', 100, 500, 'int'], 
            ['lr', .001, .01, 'float']
            ]
        >>> fixed_var = [
            ['seed', 0]
            ]
        >>> c = util.Calibrate_Mean_NN(BO_var, fixed_var)
        >>> c.run(manager, num_trials = 20, num_rec_points = 3)
    """
    def __init__(self, BO_var, fixed_var, res_filename):
        self.layers = []
        super().__init__(BO_var, fixed_var, res_filename)
    
    def _translate_vars(self, variables):
        layers = self.layers.copy()
        epoch = self.epoch
        lr = self.lr
        seed = self.seed
        for x, r in zip(variables, self.BO_var):
            if (r[0] == 'layer') & (x>= int(r[1])):
                layers.append(int(round(x)))
            if (r[0] == 'lr'):
                lr = float(np.clip(x, r[1], r[2]))
            if (r[0] == 'epoch'):
                epoch = int(np.clip(round(x), r[1], r[2]))
            if (r[0] == 'seed'):
                seed = float(np.clip(x, r[1], r[2]))
        return layers, lr, epoch, seed


    def run(self, manager, DR_model, num_grid_points = 2000, num_trials = 30, num_rec_points = 3, acq_type = 'EI', quiet = True):
        pinfo = self.set_base(manager)
        pinfo.update_parameter('num_grid_points', num_grid_points)
        pinfo.update_parameter('num_rec_points', num_rec_points)
        pinfo.update_parameter('acq_type', acq_type)
        #step 2: use BO on NN_calb.txt over NN variables
        for l in range(0, num_trials+1):
            #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
            _, pend_samples = archiver.import_dataset(self._res_filepath)
            util.thread_it(self.eval_loss, [(manager, DR_model, row, quiet) for row in pend_samples])
            if l<num_trials:
            #If less then the number of trials we run, run another Bayes set
                print("running loop number %d of %d" % (l+1, num_trials))
                bo.main_loop(pinfo)
        return self.best_solution()

    def eval_loss(self, manager, DR_model, variables, quiet = True):
        layers_m, learning_rate_m, epochs_m, seed = self._translate_vars(variables)
        count = min(0, len(layers_m)-1)
        NN_mean_var = [
            manager.dim_in, 
            manager.dim_out, 
            epochs_m, learning_rate_m, layers_m
            ]
        torch.random.manual_seed(seed)
        mean_model = cgp.Mean_NN(NN_mean_var, DR_model) #not the NN inputs
        #NN_mean_var = [dim_in, dim_out, epochs_m, learning_rate_m, layers_m]
        mean_model.calculate(manager, quiet = quiet)
        train_X, train_Y = manager.load_training()
        X_0 = transforms.normalize(train_X, manager.orig_range[0])
        inputs, labels = torch.as_tensor(X_0), torch.as_tensor(train_Y)
        y_hat = mean_model.NN_func(inputs)
        loss = mean_model.NN_func.error_function(labels, y_hat)
        #save these in the original results file we keep
        y = loss.item() + (max(1, loss.item*.1)*count)
        if not quiet:
            print(y)
        archiver.update_record([variables], ["objective"], [[y]], self._res_filepath)











# def find_dimDR(manager, DR_range):
#     _, _, seed, NN_v, _ = archiver.load_DR_settings(manager._settings_filename)
#             #[epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
#     NN_var = [
#         manager.dim_in, 
#         manager.dim_out, 
#         *NN_v
#         ]
        
#         util.thread_it(Test_DRNN_Arch, [(manager, seed, manager.orig_range, dim_DR, NN_var) for range(DR_range)manager, row, quiet) for row in pend_samples])

#     for dim_DR in range(DR_range[0], DR_range[1]):
#         [[dim_DR, Test_DRNN_Arch(manager, seed, manager.orig_range, dim_DR, NN_var)] for range(DR_range)


# def Test_DRNN_Arch(manager, seed, orig_range, dim_DR, NN_var, quiet = False):
#     #NN_var = [dim_in, dim_out, epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
#     torch.random.manual_seed(seed)
#     tDR_model = dim_red.NN_method(dim_DR, orig_range, NN_var) #orig range = not the Nn inputs
#         #NN_var = [dim_in, dim_out, epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
#     train_X, train_Y = manager.load_training()
#     tDR_model.calculate(train_X, train_Y, quiet = quiet)
#     X_0 = tDR_model.prep_X(train_X)
#     inputs, labels = torch.as_tensor(X_0), torch.as_tensor(train_Y)
#     _, y_hat, x_hat = tDR_model.Model(inputs)
#     loss = tDR_model.Model.error_function(
#             x_hat.detach(), y_hat.detach(), 
#             inputs, labels, 
#             tDR_model.norm_range[:, 0], 
#             tDR_model.norm_range[:, 1]
#             )
#     #save these in the original results file we keep
#     y = loss.item()
#     if not quiet:
#         print(y)
#         archiver.record_eval((variables, y, self.res_filename)
    
