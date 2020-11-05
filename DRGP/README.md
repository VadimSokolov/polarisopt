POLARIS Calibration Methods
===========
Execution code for Bayesian optimization algorithms for calibration of transportation simulators in distributed computing environments with multiple techniques for pre-processing to reduce dimensionality.

Given a set of variables that should be, a set of samples are recommended and evaluated in order to learn as much as possible about the relationship between the value of these variables and the output of the simulator in relation to a provided goal (for example, we may be trying to calibrate the variables to provide an output which matches as closely as possible to field-collected values for travel times)

Reduction Methods 
-----------------
the code will first create a reduced dimensionality set over original variables identified using the specified technique and then run the Bayesian optimization method on a GP built over the reduced subspace

* *Principal Component Analysis* (PCA)
* *Partial Least Squares* (PLS)
* *Active Subspace* (AS)
* *Neural Network* (NN)

Additionally, an NN mean m(x) can be created for any GP(m(x),K(x,x')) constructed by the code using a Neural Network


Required Packages
-----------------
* Python 3.6
* **[Pytorch python package](https://pytorch.org/)**
* **[GPytorch python package](https://gpytorch.ai/)**
* **[Botorch python package](https://botorch.org/)**
* **[Active Subspaces package](http://activesubspaces.org/)**
* **[scikit-learn]** - for PCA and PLS methods
* threading module for parallelization
* numpy, scipy, and matplotlib python packages

