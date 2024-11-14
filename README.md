
POLARIS Variable Calibration
===========
This package executes code for calibration and exploration of POLARIS static variables such as those found in ```*DestinationChoiceModel.json ```. It combines Bayesian Optimization (BO), parallel evaluation, and Dimension Reduction (DR) techniques to handle large variable sets with few samples. Current DR techniques include:

* *Principal Component Analysis* (PCA)
* *Partial Least Squares* (PLS)
* *Active Subspace* (AS)
* *Neural Network* (NN)

Additionally, a NN mean m(x) can be created for any GP(m(x),K(x,x')) constructed and subspaces can be made to update at regular intervals during the BO

## Requirements
## File Structure
This program assumes the following structure:

* ```data``` folder
    * ```.json``` configuration file
    * ```.json``` settings file
*```PolarisOpt``` folder containing this package's code
*```simulator``` folder
    * ```POLARIS``` folder containing the necessary information to run the simulator
    * ```Target``` folder containing the outputs calibration is seeking to match


## Packages
-----------------
* Python 3.6
* **[Pytorch python package](https://pytorch.org/)**
* **[GPytorch python package](https://gpytorch.ai/)**
* **[Botorch python package](https://botorch.org/)**
* **[Active Subspaces package](http://activesubspaces.org/)**
* **[scikit-learn]** - for PCA and PLS methods
* threading module for parallelization
* numpy, scipy, and matplotlib python packages


## Paper
[Dropbox](https://www.dropbox.com/s/50c4stluokhyycr/polaris-drbo.pdf?dl=1)
The papeer was submitted to Arxiv on March 8, 2022. Will update when published

## Using POLARIS on Bebop:
https://github.com/anl-polaris/polaris-hpc/wiki/Using-POLARIS-on-Bebop

# Workflow

# PolarisOpt Package tutorial

The outline of this tutorial is as follows:

* Creating the parameter manager
* Build and/or evaluate pending samples using the simulator
* Building the models governing dimension reduction and a potential NN mean for the GP
* Calibration of a simulator

## Step 1: Create a Managing Class

```setup_manager.SetupManager()```

This is the class used to create a central repository of parameter information to run all main functions. To create an instance, the class requires the following inputs:
 * a ```settings.json``` file defining the scenario control parameters. Must be located in the ```data``` folder. For details, see ```data\settings_readme.md```
 * a ```config.json``` file defining the POLARIS calibration variables. Must be located in the ```data``` folder. For details, see ```data\config_readme.md```


```python

from PolarisOpt.setup_manager import SetupManager
settings_filepath = "settings.json"
config_filepath = "config.json"
manager = SetupManager(settings_filepath, config_filepath)
```

## Step 2: Building Datasets

In order to run the calibration and dimension reduction functions, a ```.json``` file, referenced by the assigned ```training_filename``` parameter in the ```settings.json```, must contain at least 1 evaluated sample.

```PolarisOpt.F.build_sampleset()```

This function, if applicable, creates a Latin Hypercube set of samples over the entire statespace of calibration variables defined in ```config_filepath``` and evaluate all pending samples in parallel batches. To run, the function requires:
 * ```manager``` (SetupManager class): the managing class
 * ```save_filename``` (file path): the file that the generated or pending samples are kept
 * ```max_parallel``` (int): the maximum number of evaluations to be performed simultaneously (Default 2)
 * ```num_samples``` (int): the number of samples which should be generated using a Latin Hypercube sampling method. If this value is set to ```0```, then the function will not generate new points and only evaluate samples not designated with ```status: Completed``` (Default 0)

```python
from PolarisOpt.F import build_sampleset
build_sampleset(manager, manager.training_filename, max_parallel = 32, num_samples=6)
```

## Step 3: Build or load the needed models for Calibration

### Build from scratch
```PolarisOpt.F.build_calibration()```

This function is used to construct and train the dimension reduction ```method``` and, if applicable, construct and train a NN for the GP mean. To run, the function requires:
* ```manager``` (SetupManager class): the managing class
* ```quiet``` (Boolean): whether or not to print any training statuses printed by the chosen models (Default ```True```)
 
    
```python
from PolarisOpt.F import build_calibration
DR_model, M_model = build_calibration(manager, quiet = False)
```
### Load from saved
```PolarisOpt.utils.archiver.load_model()```

This function allows the user to alternatively load a previously stored model. To run, the function requires:
* ```filename``` (path): the full path to the ```.pickle``` file containing the previous-trained model

"""
```python
from PolarisOpt.utils.archiver import load_model
DR_model = load_model("C:\\POLARIS_Runs\\data\\Models\\AS_model.pickle")
M_model = None
```

## Step 4: Run Calibration
   ```PolarisOpt.F.calibrate_simulation()```
This is the function used to run the Bayesian Optimization for ```num_BO_loops``` iterations with ```num_BO_rec``` recommendations. Pending evaluations are evaluated in parallel. To run, the function requires:
* ```manager``` (SetupManager class): the managing class
* ```DR_model``` (DR_technqiue class): the dimension reduction method class
* ```M_model``` (None or Mean_NN class): ```None``` if a NN mean for the GP is not desired; otherwise, the NN class (Default ```None```)
* ```quiet``` (Boolean): whether (```False```) or not to (```True```) print any training statuses printed by the chosen models (Default ```True```)
 
```python
from PolarisOpt.F import calibrate_simulation
calibrate_simulation(manager, DR_model, M_model, quiet = 'False')
````

