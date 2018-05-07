Milestone 1
===========


BACKGROUND
------

This folder contains the resources used for the first milestone of the 517a application project. 

The goal of this milestone was to create a linear regression model for the selected dataset that would predict the number of times an article was shared using a variety of features. [Check out the wiki for more information about the procedure and results.](https://github.com/jlandman/CSE517A-Application-Project/wiki/Milestone-1)

CODE
------

All .py files can be run independently using Python 3.
Libraries used include numpy, scikit-learn, and matplotlib

Required

The file target_comparison.py generates histograms that show the rationale behind estimating the logarithm of the number of shares, rather than the number itself.

The file ridge_10cv_scan.py performs 10-fold cross validation on the data using a ridge regression model. Various values of lambda are tested with the same (randomly shuffled) dataset. The cross-validation error is minimized over possible values of lambda, and both the training and cross-validation error are plotted as a function of lambda.

The file ridge_10cv_miniminze.py performs the same 10-fold cross validation, except instead of scanning lambda values, lambda is minimized over the continuously shuffled dataset. This minimization is done with a simulated annealing process (to deal with uncertainty), and the results are also plotted.

The file feature_plotter.py trains a ridge regression model (with the lambda mentioned above) and breaks down the impact of the various types of features used.

The file model_evaluator.py trains a ridge regression model on a randomly selected training and test set (80% training, 20% test) 100 times to generate average in and out of sample mean error, mean squared error, and R^2 values.


