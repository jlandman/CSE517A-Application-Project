Milestone 2
===========

BACKGROUND
------

This folder contains the resources used for the second milestone of the 517a application project. 

The goal of this milestone was to train and run Gaussian Processes and evaluate and compare the predictions using at least two differnt kernels via 10-fold cross-validation with a suitable error measure. Check out the wiki for more information about the procedure and results.

CODE
------

All .R files can be run using R version 3.4.3
Packages used include caret, kernlab, dplyr, and plot3D

The file data_prep.R reads in the data and splits it into two training sets (one to learn the best kernel parameter values and one to learn the best overall model) and a training set. It also splits the training sets into folds for cross validation.

The file nlpd.R creates a function to calculate the negative log predictive density for use as an error measure.

The file polynomial.R uses 10-fold cross validation to find the best parameter values for Gaussian Processes using the polynomial kernel.

The file rbf.R uses 10-fold cross validation to find the best parameter value for Gaussian Processes using the rbf kernel.

The file best.R uses 10-fold cross validation to compare Gaussian Processes using the linear kernel, th epolynomial kernel with parameter values found using polynomial.R and the rbf kernel with parameter value found using rbf.R.

The file main.R was written to run each of the above files in succession. This is the file that should be run to repeat the current analysis using the above helper files.
