from abc import ABC, abstractmethod
import os
import sys
import torch
import dill
import math
from . import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from . import eval_sim
from .utils import archiver
from .utils import transforms
from .utils.objective_funcs import run_objective
import shutil
try:
    import active_subspaces as ac
except ImportError:
    print('Active Subspace dimension reduction technique is unavailable; Active Subspace package could not be loaded')
try:
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LinearRegression
except ImportError:
    print ('Principal Componenet Analysis and Partial Least Squares dimension reduction techniques are unavailable; sklearn package could not be loaded')

torch.set_default_tensor_type(torch.DoubleTensor)
@ -92,7 +92,7 @@ def create_DR(manager, training_file = '', quiet = False):
    r"""Main function to create dimension reduction model and accompanying information

    Args:
        manager (SetupManager class): contains the details of the original problem and dataset
        training_file (path): full path to the sample set used to train the subspace on
        quiet (boolean): to (True) or not to (False) allow any subspace technique printouts
    Returns:
        dimension reduction model (class)
    """
    if training_file == '':
        training_file=manager.training_filename
        
    ############################################
    #Step 1: Establish needed info             #
    ############################################
    method, dim_DR, seed_value, NN_var, _ = archiver.load_DR_settings(manager.settings_filename)
    torch.random.manual_seed(seed_value)
    #we automatically save the generated model in the same location as the results file given
    if not os.path.exists(training_file):
        ValueError('%s does not exist' % training_file)
    m_folder = os.path.join(os.path.dirname(manager.res_filename), 'Models')   #automatically saved in the results file folder
    if not os.path.exists(m_folder):
        os.mkdir(m_folder)
    model_fn = os.path.join(m_folder, manager.res_model_filename)

    ############################################
    #Step 1: Create instance of technique model#
    ############################################
    if method=='None':
        DR_model = No_method(manager.orig_range)
    elif method=='PCA':
        DR_model = PCA_method(dim_DR, manager.orig_range)
    elif method=='PLS':
        DR_model = PLS_method(dim_DR, manager.orig_range)                
    elif method=='AS':
        DR_model = AS_method(dim_DR, manager.orig_range)
    elif method=='NN':
        DR_model = NN_method(dim_DR, manager.orig_range, [manager.dim_in, manager.dim_out, *NN_var])
        #NN_var = [dim_in, dim_out, epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
    else:
        raise ValueError('Method not valid')

    ############################################
    #Step 2: read in and set up training file  #
    ############################################

    train, _ = archiver.import_dataset(training_file, x_key = "orig_input", y_key = "target_err")
    if train.shape[1] != (manager.dim_in + manager.dim_out):
        raise ValueError('Expected %s columns but got %s' % ((manager.dim_in + manager.dim_out), train.shape[1]))
    train_X, train_Y_err = train[:, manager.dim_out:], train[:, :manager.dim_out]
    train_Y_obj, _ = run_objective(train_Y_err,manager.objective_type)
    
   ############################################
    #Step 3: Run the class' funct to train     #
    ############################################
    if method == 'NN':
        DR_model.calculate(train_X, train_Y_err, quiet= quiet)
    else:
        DR_model.calculate(train_X, train_Y_obj)
    train_DR = DR_model.encode_X(train_X)

    ##################################################
    #Step 4: save results in results file used by GPs#
    ##################################################
    shutil.copyfile(training_file, manager.res_filename)
    archiver.update_record(train_X, ["DR_input","objective"], list(zip(train_DR, train_Y_obj)), manager.res_filename)
    _ = archiver.save_model(DR_model, model_fn)
    return DR_model




def tune_DR(manager, quiet=False):
    r"""Main function to update dimension reduction model and accompanying information

    Args:
        manager (archiver.SetupManager class): contains the details of the original problem and dataset
    Returns:
        dimension reduction model (class)
    """
    return create_DR(manager, training_file = manager.res_filename, quiet = quiet)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

class DR_Technique(ABC):
    r"""Abstract base class for dimension reduction techniques.
        """
    def __init__(self, method, dim_DR, orig_range):
        r"""Constructor for the base class of techniques

        Args:
            dim_DR (int): number of dimensions to reduce to
            orig_range (ndarray): the [lower, upper] bounds of the original subspace
        """
        super().__init__()
        self.method = method
        self.dim_DR = dim_DR
        self.orig_range = orig_range
        self.DR_range = None
        
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
    @abstractmethod
    def calculate(self, train_X, train_Y):
        r"""returns a dim red model
        """
        pass

    @abstractmethod
    def decode_X(self, DR_input):
        r"""takes a reduced input and returns estimate x in original subspace
        """
        pass

    @abstractmethod
    def encode_X(self, x_input):
        r"""takes a original-subspace input and returns the dim reduced equivalent
        """
        pass

    @abstractmethod
    def pred_Y(self, DR_input):
        r"""takes a reduced input and returns estimate y in original subspace
        """
        pass

    def enforce_bounds(self, x_input):
        r"""takes a original-scaled input and returns the array with bounds enforced and corrects for integer requirements
        """
        for i in range(0, x_input.shape[-1]):
            if self.orig_range[1][i]=='int':
                x_input[:, i] = int(round(x_input[:, i]))
        return np.clip(x_input, self.orig_range[0][:, 0], self.orig_range[0][:, 1])



##########################################################################################################
##########################################################################################################
#                                    Default No Reduction Method                                         #
##########################################################################################################
##########################################################################################################

class No_method(DR_Technique):
    r"""a placeholder for when no reduction is made for ease of use

    This simply returns information provided

    Example:
        >>> DR_model = No_method(0, [0, 1])
        >>> DR_model.calculate(train_X, train_Y)
    """
    def __init__(self, orig_range):
        r"""Args:
            orig_range: the bounds of the original subspace
        """
        super().__init__('None', None, orig_range)
        self.DR_range = self.orig_range[0]
        self.Model =  None

    def calculate(self, train_X, train_Y):       
        return print('Placeholder created')

    def decode_X(self, DR_input):
        return self.enforce_bounds(DR_input)

    def encode_X(self, x_set):
        return x_set

    def pred_Y(self, DR_set):
        raise ValueError("No Reduction Method used; cannot predict Y")


##########################################################################################################
##########################################################################################################
#                                     Partial Least Squares Method                                      #
##########################################################################################################
##########################################################################################################

class PLS_method(DR_Technique):
    r"""Partial Least Squares dimension reduced subspace

    This computes reduced subspace by:
    (1) standardizing x and y
    (1) applying 2-blocks regression PLS2 over x and y

    Example:
        >>> DR_model = PLS_method(0, [0, 1])
        >>> DR_model.calculate(train_X, train_Y)
    """
    def __init__(self, dim_DR, orig_range):
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace and their types (float or int)
        """

        if dim_DR>0 and isinstance(dim_DR, int):
            super().__init__('PLS', dim_DR, orig_range)
        else:
            raise ValueError('dim_DR must be an integer greater than 0')
        
    def calculate(self, train_X, train_Y):
        self.Model = PLSRegression(n_components = self.dim_DR)   #automatically standardizes everything for us
        self.Model.fit(train_X, train_Y)
        DR = self.Model.transform(train_X) #will standardize for us automatically
        self.x_std = self.Model.x_std_
        self.x_mean = self.Model.x_mean_
        self.y_std = self.Model.y_std_
        self.y_mean = self.Model.y_mean_        
        self.DR_range = np.c_[np.min(DR, axis=0)*1.1, np.max(DR, axis=0)*1.1] #pad a bit
        return print('PLS model created')

    def decode_X(self, DR_input):
        xhat_n = np.dot(DR_input, self.Model.x_rotations_.T)
        xhat = transforms.inverse_standardize(xhat_n, self.x_mean, self.x_std)
        return self.enforce_bounds(xhat)

    def encode_X(self, x_set):
        X_0 = transforms.standardize(x_set, self.x_mean, self.x_std)
        return np.dot(X_0, self.Model.x_weights_)

    def pred_Y(self, DR_set):
        #Y = TQ'+F
        yhat_n = np.dot(DR_set, self.Model.y_loadings_.T)
        if len(yhat_n)==1:
            yhat_n = yhat_n[:, None]
        if self.Model.norm_y_weights:
            return transforms.inverse_standardize(yhat_n, self.y_mean, self.y_std)
        else:
            return yhat_n

##########################################################################################################
##########################################################################################################
#                                   Principal Component Analysis Method                                  #
##########################################################################################################
##########################################################################################################

class PCA_method(DR_Technique):
    r"""Principal Component Analysis dimension reduced subspace

    This computes reduced subspace by:
    (1) standardizes the inputs
    (2) applies PCA 
    (3) applies a Linear Regression over the PCA subspace and Y since this is ignored by PCA

    Example:
        >>> DR_model = PCA_method(0, [0, 1])
        >>> DR_model.calculate(train_X, train_Y)
    """
    def __init__(self, dim_DR, orig_range):  
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace
        """

        if dim_DR>=0 and isinstance(dim_DR, int):
            super().__init__('PCA', dim_DR, orig_range)
        else:
            raise ValueError('dim_DR must be an integer greater than or equal to 0')      
        self.Modely = None
        self.Model = None

    def calculate(self, train_X, train_Y):
        if np.size(train_Y, axis=1)!=1:
            raise ValueError('Y cannot consist of more than 1 variable or is not a 2D array')
        
        #####################################################
        #Step 1: standardize X                                #
        #####################################################
        self.x_mean, self.x_std=transforms.calc_stats(train_X)
        self.y_mean, self.y_std=transforms.calc_stats(train_Y)
        X_0 = transforms.standardize(train_X, self.x_mean, self.x_std)

        #####################################################
        #Step 2: Find PCA for DR info                       #
        #####################################################  
        if self.dim_DR==0: #if the user chose to interactively choose dim_DR
            self.Model = PCA().fit(X_0)
            plt.plot(np.cumsum(self.Model.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Explained Variance (%)') #for each component
            plt.show()
            #prompt choice of dim_DR
            self.dim_DR = int(input("what number of dimensions do you want?(>0):"))
        self.Model = PCA(n_components = self.dim_DR)
        self.Model.fit(X_0)
        
        #####################################################
        #Step 4: Learn new Dim range                        #
        #####################################################
        DR = self.Model.transform(X_0)
        self.DR_range = np.c_[np.min(DR, axis=0)*1.1, np.max(DR, axis=0)*1.1] #pad a bit

        #####################################################
        #Step 5: Train y on Lower Dimension Set             #
        #####################################################
        self.Modely = LinearRegression()
        self.Modely.fit(DR, train_Y)
        return print('PCA model created')

    def decode_X(self, DR_input):
        if len(DR_input.shape)==1:
            DR_input=DR_input[None, :]
        xhat = self.Model.inverse_transform(DR_input)
        xhat = transforms.inverse_standardize(xhat, self.x_mean, self.x_std)
        return self.enforce_bounds(xhat)

    def encode_X(self, x_set):
        X_0 = transforms.standardize(x_set, self.x_mean, self.x_std)
        return self.Model.transform(X_0)

    def pred_Y(self, DR_input):
        if len(DR_input.shape)==1:
            DR_input=DR_input[None, :]
        return self.Modely.predict(DR_input)


##########################################################################################################
##########################################################################################################
#                                           Active Subspaces Method                                      #
##########################################################################################################
##########################################################################################################


class AS_method(DR_Technique):
    r"""Active Subspace dimension reduced subspace

    This computes reduced subspace by:
    (1) normalizing between [-1, 1]
    (2) estimating the derivatives locally
    (3) learns the inactive and active subspaces
    (4) applies radial basis regression from reduced dimension to y 

    Example:
        >>> DR_model = AS_method(0, [0, 1])
        >>> DR_model.calculate(train_X, train_Y)
    """
    def __init__(self, dim_DR, orig_range):  
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace and their types (float or int)
        """
        if dim_DR>=0 and isinstance(dim_DR, int):
            super().__init__('AS', dim_DR, orig_range)
        else:
            raise ValueError('dim_DR must be an integer greater than or equal to 0')
        self.norm_range = None
        self.W1 = None
        self.W2 = None
        self.map = None

    def calculate(self, train_X, train_Y):
        #####################################################
        #Step 1: normalize data                             #
        #####################################################
        if len(train_X)<(2*len(self.orig_range[0])):
            ValueError('Active Subspace reduction requires a training set sample size of at least %d' % (len(self.orig_range[0])))
        XX = transforms.normalize(train_X, self.orig_range[0])
        X_0 = (2*XX)-1 #AS requires [-1, 1] range
        self.norm_range = np.c_[-np.ones(self.orig_range[0].shape[0]), np.ones(self.orig_range[0].shape[0])]
        
        #####################################################
        #Step 2: Find AS for DR info                        #
        #####################################################  
        if self.dim_DR==0:
            #interactive dim finder
            ss = ac.subspaces.Subspaces()
            df8 = ac.gradients.local_linear_gradients(X_0, train_Y)
            ss.compute(df = df8, nboot = 1000)
            self.dim_DR = np.shape(ss.W1)[1]
            ###visualize findings + adjust W1 if needed
            ac.utils.plotters.eigenvalues(ss.eigenvals[0:10], ss.e_br[0:10])
            ac.utils.plotters.subspace_errors(ss.sub_br[0:10])
            if np.shape(ss.W1)[1]==1:
                ss.partition(2)
                ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), train_Y[:, 0])
                answer = input("Do we include second dimension in active subspace (Y or N):")
                if answer=="Y":
                    self.dim_DR = 2       
        ss = ac.subspaces.Subspaces()
        df8 = ac.gradients.local_linear_gradients(X_0, train_Y)
        ss.compute(df = df8, nboot = 1000)
        ss.partition(self.dim_DR)
        self.W1 = ss.W1
        self.W2 = ss.W2

        #####################################################
        #Step 3: Learn new Dim range                        #
        #####################################################
        DR = X_0.dot(self.W1)
        self.DR_range = np.c_[np.min(DR, axis=0)*1.1, np.max(DR, axis=0)*1.1] #pad a bit

        #####################################################
        #Step 4: Train y on Lower Dimension Set             #
        #####################################################
        avmap = ac.domains.BoundedActiveVariableDomain(ss)
        self.map = ac.domains.BoundedActiveVariableMap(avmap)
        ##this output is weird? example has [-1.03, 1.03] but data converted shows [-1.]
        self.Modely = ac.response_surfaces.ActiveSubspaceResponseSurface(self.map)
        self.Modely._train(DR, train_Y)
        return print('AS model created')
               
    # def reconstruct_x(self, DR_input):

    #     [4x5] = [5x2][] + [5x3][]
    #     #x = W1y + W2*z
    #     w1y = np.dot(DR_input, self.W1.T)
    #     #W2z lowerbound
    #     W2zlb = self.norm_range[:, 0]-w1y
    #     W2zub = self.norm_range[:, 1]-w1y
    #     z_shape=[len(DR_input), self.W2.shape[1]]

    #     def objective(z):
    #         return np.sum(np.square(z))

    #     constraints = []
    #     for i in range(z_shape[0]):
    #         def f(z, i = i):
    #             return np.dot(z[i*z_shape[1]:(i+1)*z_shape[1]],self.W2.T) - W2zlb[i]
    #         constraints.append(f)

    #     for i in range(z_shape[0]):
    #         def b(z, i = i):
    #             return W2zub[i]-np.dot(z[i*z_shape[1]:(i+1)*z_shape[1]],self.W2.T)
    #         constraints.append(b)
    #     constraints1 =  [{'type': 'ineq', 'fun': cons} for cons in constraints]
    #     z = optimize.minimize(objective, np.zeros(np.prod(z_shape)),constraints = constraints1)

    #     return np.dot(DR_input, self.W1.T) + np.dot(z.x.reshape(*z_shape), self.W2.T)

    def decode_X(self, DR_input):
        if len(DR_input.shape)==1:
            DR_input=DR_input[None, :]
        xhat, _ = self.map.inverse(DR_input)
        xhat = ((xhat+1)*(self.orig_range[0][:, 1]-self.orig_range[0][:, 0])/2) + self.orig_range[0][:, 0]
        # #if a batch set, we have to loop through the 2D subsets with opt
        # if len(DR_input.shape)==3:
        #     xhat = np.empty((DR_input.shape[0], DR_input.shape[1], len(self.orig_range[0])))
        #     for i in range(0, DR_input.shape[0]):
        #         xhat[i, :, :] = self.reconstruct_x(DR_input[i])
        # else:
        #     if len(DR_input.shape) == 1:
        #         DR_input=DR_input[None,:]
        #     xhat = np.empty((DR_input.shape[0], len(self.orig_range[0])))
        #     for i in range(0,DR_input.shape[0]):
        #         xhat[i,:] = self.reconstruct_x(DR_input[i:i+1,:])
        # xhat = ((xhat+1)*(self.orig_range[0][:, 1]-self.orig_range[0][:, 0])/2) + self.orig_range[0][:, 0]
        return self.enforce_bounds(xhat)

    def encode_X(self, x_set):
        XX = transforms.normalize(x_set, self.orig_range[0])
        X_0 = (2*XX)-1 #AS requires [-1, 1] range
        DR, _= self.map.forward(X_0)
        return DR

    def pred_Y(self, DR_input):
        if len(DR_input.shape)==1:
            DR_input=DR_input[None, :]
        return self.Modely.predict_av(DR_input)[0]


##########################################################################################################
##########################################################################################################
#                                         Neural Networks Method                                         #
##########################################################################################################
##########################################################################################################


class NN_method(DR_Technique):
    r"""Deep Neural Network dimension reduced subspace

    This computes reduced subspace by:
    (1) normalizes the inputs
    (2) applies a constructed NN which learns how to translate to the reduce dimension subspace
    while taking into account the X-Y relationship and the necessity to translate from reduced subspace
    to the original subspace

    Example:
        >>> DR_model = NN_method(0, [0, 1], NN_var)
        >>> DR_model.calculate(train_X, train_Y)
    """
    def __init__(self, dim_DR, orig_range, NN_var, pref_cpu = True):
        r"""Args:
            dim_DR: number of dimensions to reduce to
            orig_range: the bounds of the original subspace and their types (float or int)
            NN_var: the necessary variables to construct the NN pulled from settings files
                dim_in: number of input dimensions
                dim_out: number of output dimensions (can be >1)
                epochs: number of loops to do for training
                learning_rate: learning rate for training of NN on training set
                lamda: lambda value weighing the importance of reconstruction vs prediction
                penalty: penalty value for exceeding the bounds of the original subspace when reconstructing the inputs
                XDR_layer: the number of nodes per layer between the inputs and the reduction layer, ex [10, 100, 10]
                DRX_layer: the number of nodes per layer between the reduction and the reconstructed input layer, ex [10, 100, 10]
                DRY_layer: the number of nodes per layer between the reduction and the output layer, ex [10, 100, 10]
                
        """

        if dim_DR>0 and isinstance(dim_DR, int):
            super().__init__('NN', dim_DR, orig_range)
        else:
            raise ValueError('dim_DR must be an integer greater than 0')
        self.norm_range = None
        self.NN_var = NN_var
        
        if torch.cuda.is_available() and not pref_cpu:
            self.device = 'cuda'
            print("Running on the GPU")
        else:
            self.device = 'cpu'
            print("Running on the CPU")

        self.Model = nn.DR_Network(self.dim_DR, self.NN_var).to(self.device)
        

    def calculate(self, train_X, train_Y, quiet=False):
        #####################################################
        #Step 1: assumes must normalize X                   #
        #####################################################
        X_0 = transforms.normalize(train_X, self.orig_range[0])
        self.norm_range = np.c_[np.zeros(train_X.shape[1]), np.ones(train_X.shape[1])]

        #####################################################
        #Step 2: create Model and train                     #
        #####################################################
        nn.train_DRNN(X_0, train_Y, self.NN_var[2], self.Model, self.norm_range, quiet)
        self.DR_range = self.Model.DR_range
        return print('NN model created')

    def decode_X(self, DR_input):
        DR_input = torch.as_tensor(DR_input, device = self.device)    
        xhat_NN = self.Model.pred_X(DR_input)
        xhat = transforms.inverse_normalize(xhat_NN.detach().cpu().numpy(), self.orig_range[0])
        return self.enforce_bounds(xhat)

    def encode_X(self, x_set):
        X_0 = torch.as_tensor(transforms.normalize(x_set, self.orig_range[0]), device = self.device)
        DR = self.Model.reduce_X(X_0)
        return DR.detach().cpu().numpy()
    
    def pred_Y(self, DR_input):
        DR_input = torch.as_tensor(DR_input, device = self.device)    
        yhat = self.Model.pred_Y(DR_input)
        return yhat.detach().cpu().numpy()

    def prep_X(self, x_set):
        inputs = transforms.normalize(x_set, self.orig_range[0])
        return torch.as_tensor(inputs, device=self.device)
