Milestone 1
===========


BACKGROUND
------

This folder contains the resources used for the first milestone of the 517a application project. 

The goal of this milestone was to create a linear regression model for the selected dataset that would predict the number of times an article was shared using a variety of features. Check out the wiki for more information about the procedure and results.

CODE
------

All files can be run independently using Python 3.

The file target_comparison.py generates histograms that show the rationale behind estimating the logarithm of the number of shares, rather than the number itself.

The file ridge_cross_validation.py can performs 10-fold cross validation on the data using a ridge regression model. The cross-validation error is minimized over possible values of lambda, and both the training and cross-validation error are plotted as a function of lambda.

The file ?.py trains a ridge regression model (with the lambda mentioned above) and breaks down the efficacy of the various types of features used.

