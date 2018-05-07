Milestone 1
===========


BACKGROUND
------

This folder contains the resources used for the fourth milestone of the 517a application project. 

The goal of this milestone was to compare the accuracy of the various models studied this semester for both training and testing. These regression models were applied to the same dataset in order to predict the number of times an article was shared using a variety of features including the subject, weekday published, sentiment, etc. [Check out the wiki for more information about the procedure and results.](https://github.com/jlandman/CSE517A-Application-Project/wiki/Milestone-4)

CODE
------

The recorded mean errors of the various trials are found in the folder error_data

The file plot_errors.py generates the plots in the graphs directory using the error data. It also performs t-tests to determine if any of the numtimes are significantly faster.

Libraries used include numpy, scikit-learn, and matplotlib



