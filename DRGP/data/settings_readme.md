A settings ```json``` file is used to define the static parameters necessary to run ```PolarisOpt``` functions. Below outlines the information that must be provided to each control parameter section.


### General simulation controls

* ```simulation_path``` (path): full path pointing to the folder of the simulator's executable and accompanying files. Filenames defined in the ```config.json``` file are assumed to be located using this path
* ```simulation_scenario_name``` (filename): the name of the ```scenario.json``` file required to run the POLARIS executable
* ```target_output_filename``` (filename):  the name of the file located in the ```simulator\Target``` folder which contains the simulation outputs calibration seeks to match
* ```output_SQL_query```(text): the SQL query that should be run on the ```target_output_filename``` and the evaluated simulation outputs for use in the objective function. Assumed to have 2 columns with the 2nd column recording the difference between the target and evaluated outputs

### File controls
* ```training_filename``` (filename): a ```.json``` file name which contains an initial sample set of evaluated points in the following format:

```json
[
    {
        "status": "Pending",
        "variables": "variable_1 variable_2...",
        "orig_input": "X1 X2...",
        "DR_input": "",
        "target_err": "Y1 Y2...",
        "objective": "P",
        "run_time": "P"
    }
]
```
These points are used for training the subspace projections and as a starting basis for the Bayesian Optimization algorithm; it must contain at least 1 point for Bayesian Optimization to run. To run ```Active Subspace``` dimension reduction method, must contain more than twice the number of variables being calibrated.
* ```res_filename``` (filename): a ```.json``` file name for where all samples resulting from the Bayesian Optimization algorithm should be stored in the ```data``` folder. Samples which have yet to be evaluated are designated with a ```status: "Pending"``` until the objective is calculated


### General BO controls

* ```num_BO_loops``` (int): the number of times the Bayesian Optimization algorithm should be looped through. A loop is defined as a recommendation cycle and an evaluation cycle (Default 1)
* ```num_rec_points``` (int): the number of samples which should be recommended for future evaluation during a single recommendation cycle of the Bayesian Optimization algorithm (Default 1)

NOTES: Unless additional pending values exist in the ```res_filename``` file, this entry will equal the number of simultaneous evaluations

* ```num_grid_points``` (int): the number of potential samples that should be considered for recommendation during a single recommendation cycle of the Bayesian Optimization algorithm. The potential pool of samples is produced by first Latin Hypercube Sampling (LHS) the entire statespace with ```num_grid_points``` before adding an additional 50 LHS [-.001,.001] around the current optimal solution. All already-evaluated or pending samples are excluded for consideration. (Default 2000)

* ```acq_type``` (text): the acquisition function to be run during the recommendation cycle of the Bayesian Optimzation algorithm. Current valid entries include: 

    - ```EI``` - Expected Improvement
    - ```SPE``` - Squared Predictive Error 

* ```objective_type``` (text): the identifier for the desired objective function as defined in ```PolarisOpt.utils.objective_funcs```.  Current valid entries include: 

    - ```MSE``` - Mean Squared Error

* ```epsilon_stop``` (float): the amount of distance from the optimal output is acceptable to prematurely end the Bayesian Optimization algorithm (Default 0.1)

* ```add_nn_GP_mean``` (boolean): to (```true```) or not to (```fase```) construct a neural network-based mean for the Gaussian Process used in the recommendation cycle. The network architecture is defined in the ```GP neural network mean controls``` section.


### General Dimension Reduction (DR) controls


* ```method``` (text): the dimension reduction method that should be implemented. Current valid entries include: 
    - ```None``` - no reduction method is implemented
    - ```PCA```  - Principal Componenet Analysis
    - ```PLS```  - Partial-Least Squares
    - ```AS```   - Active Subspaces
    - ```NN```   - the unique neural network outlined in <paper reference>

* ```dim_DR``` (int): the number of dimensions in the reduction layer, which become inputs to the GP. If method = ```None```, this setting is ignored. Current valid entries include:
   - If method = ```AS``` or method = ```PCA```, this value can be set to ```0``` to initiate an interactive determination. However, this parameter must be updated manually with the final decision
    - All other methods only accept a positive integer greater than 1

* ```seed``` (float): a seed value to allow for reproduceable results

* ``````method_mean_update``` (boolean): to (```true```) or not to (```fase```) retrain the dimension reduction subspace after ```method_update_interval``` loops of the algorithm

* ```method_mean_update_interval``` (int): The number of loops before the dimension reduction subspace should be retrained. This will only be applied if the ```method_update``` boolean is set to ```true```
* ```nn_mean_update``` (boolean): to (```true```) or not to (```fase```) retrain the GP's NN mean after ```nn_mean_update_interval``` loops of the algorithm. This will only be applied if the ```add_nn_GP_mean``` boolean is set to ```true```
* ```nn_mean_update_interval``` (int): The number of loops before the GP's NN mean should be retrained. This will only be applied if the ```add_nn_GP_mean``` and ```nn_mean_update``` boolean are both set to ```true```


### DR neural network controls

This section is only read when ```method = NN```.

* ```epochs_n``` (int): the number of training iterations to perform. Any positive integer is accepted; typical entries are in interval ```[300,1000]```

* ```learning_rate_n``` (float): the learning rate used during training. A positive float value with typical entries in interval ```(0,1]```

* ```lambda_n``` (float): a counter weight ```lambda``` placed on the outputs to change the network's emphasis to accurately learn outputs over inputs

* ```penalty_n``` (float): penalty weight for reconstructions of the inputs exceeding the boundries defined in the ```config.json``` file when training

* ```XDR_layer``` (list): a list of integers depicting the number of fully-connected network nodes per layer between input and the dimension-reduced layer
* ```DRX_layer``` (list): a list of integers depicting the number of fully-connected network nodes per middle layer between dimension-reduced layer and reconstructed inputs
* ```DRY_layer``` (list): a list of integers depicting the number of fully-connected network nodes per middle layer between dimension-reduced layer and predicted outputs
 

### GP neural network mean controls

This section is only read when ```add_nn_GP_mean = true``` 
    
* ```epochs_m``` (int): designates the number of training iterations performed. Any positive value greater than zero is accepted; typical entries are in interval ```[300,1000]```

* ```learning_rate_m``` (float): the learning rate used during training.Any positive float value is accepted; typical entries are in interval ```(0,1]```
                
* ```layers_m``` (list): a list of integers depicting the number of fully-connected network nodes per middle layer between input and predicted output



---------------------------------------------------------
---------------------------------------------------------
                Example config.json setup
---------------------------------------------------------
---------------------------------------------------------
```json
{
	"General simulation controls" :
	{
        "simulation_path" : "C:\\POLARIS_Runs\\simulator\\Polaris\\bloomington_model",
        "simulation_scenario_name" : "scenario.json"
    },

    "File controls" :
    {
        "training_filename" : "training_data.txt",
        "res_filename" : "NN_results.dat"
    },

    "General BO controls" :
	{
        "num_BO_loops" : 20,
        "num_rec_points" : 2,
        "num_grid_points" : 2000,
        "acq_type" : "EI",
        "epsilon_stop" : 0.1,
        "add_nn_GP_mean" : false	
    },

    "General DR controls" :
    {
        "method" : "None",
        "dim_DR" : 3,
        "seed_value" : 2,
        "method_update" : false,
        "method_update_interval" : 2,
        "nn_mean_update" : false,
        "nn_mean_update_interval" : 2
    },

    "DR neural network controls" :
    {
        "epochs_n" : 1000,
        "learning_rate_n" : 0.001,
        "lambda_n" : 0.0005,
        "penalty_n" : 200,
        "XDR_layer" : [800,700,400],
        "DRX_layer" : [1000,1000],
        "DRY_layer" : [300,900,400]
    },

    "GP neural network mean controls" :
    {
        "epochs_m" : 1000,
        "learning_rate_m" : 0.001,
        "layers_m" : [500, 100, 100]
    }
}
```