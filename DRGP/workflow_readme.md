# PolarisOpt Package tutorial

The outline of this tutorial is as follows:

* Creating the parameter manager
* Build and/or evaluate pending samples using the simulator
* Building the models governing dimension reduction and a potential NN mean for the GP
* Calibration of a simulator

## Step 1: Create a Managing Class

```setup_manager.SetupManager()```

This is the class used to create a central repository of parameter information to run all main functions. To create an instance, the class requires the following inputs:
 * a ```settings.json``` file defining the scenario control parameters. For details, see ```data\settings_readme.md```
 * a ```config.json``` file defining the POLARIS calibration variables . For details, see ```data\config_readme.md```


```python

from PolarisOpt.setup_manager import SetupManager
settings_filename = "C:\\POLARIS_Runs\\data\\settings.json"
config_filename = "C:\\POLARIS_Runs\\data\\config.json"
manager = SetupManager(settings_filename, config_filename)
```

## Step 2: Building Datasets

In order to run the calibration and dimension reduction functions, a ```.txt``` or ```.dat``` file, referenced by the assigned ```training_filename``` parameter in the ```settings.json```, must contain at least 1 evaluated sample in the format ```[Y,X]```. 
* ```Y``` represents the difference between the target scenario outputs and simulation outputs. For pending samples, ```Y``` is placeheld with ```"P"```
* ```X``` represents the settings used for the calibration variables defined in ```config_filename```

```PolarisOpt.utils.util.build_sampleset()```

This function, if applicable, creates a Latin Hypercube set of samples over the entire statespace of calibration variables defined in ```config_filename``` and evaluate all pending samples in parallel batches. To run, the function requires:
 * ```manager``` (SetupManager class): the managing class
 * ```save_filename``` (file path): the file that the generated or pending samples are kept
 * ```max_parallel``` (int): the maximum number of evaluations to be performed simultaneously (Default 2)
 * ```num_samples``` (int): the number of samples which should be generated using a Latin Hypercube sampling method. If this value is set to ```0```, then the function will not generate new points and only evaluate samples starting with ```"P"``` (Default 0)

```python
from PolarisOpt.utils.util import build_sampleset
build_sampleset(problem_info, problem_info.training_filename, max_parallel = 32, num_samples=6)
```

## Step 3: Build or load the needed models for Calibration

### Build from scratch
```PolarisOpt.utils.util.build_calibration()```

This function is used to construct and train the dimension reduction ```method``` and, if applicable, construct and train a NN for the GP mean. To run, the function requires:
* ```manager``` (SetupManager class): the managing class
* ```pr``` (Boolean): whether or not to print any training statuses printed by the chosen models (Default ```False```)
 
    
```python
from PolarisOpt.utils.util import build_calibration

DR_model, M_model = build_calibration(manager)
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
   ```PolarisOpt.utils.util.calibrate_simulation()```
This is the function used to run the Bayesian Optimization for ```num_BO_loops``` iterations with ```num_BO_rec``` recommendations. Pending evaluations are evaluated in parallel. To run, the function requires:
* ```manager``` (SetupManager class): the managing class
* ```DR_model``` (DR_technqiue class): the dimension reduction method class
* ```M_model``` (None or Mean_NN class): ```None``` if a NN mean for the GP is not desired; otherwise, the NN class (Default ```None```)
* ```pr``` (Boolean): whether or not to print any training statuses printed by the chosen models (Default ```False```)
 
```python
from PolarisOpt.utils.util import calibrate_simulation
calibrate_simulation(manager, DR_model, M_model)
````

