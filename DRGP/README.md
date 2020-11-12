
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

