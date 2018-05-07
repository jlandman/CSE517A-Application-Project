Milestone 1
===========


BACKGROUND
------

This folder contains the resources used for the ninth optional task of the 517a application project. 

The goal of this milestone was to compare the efficiency (speed) of the various models studied this semester for both training and testing. These regression models were applied to the same dataset in order to predict the number of times an article was shared using a variety of features including the subject, weekday published, sentiment, etc. [Check out the wiki for more information about the procedure and results.](https://github.com/jlandman/CSE517A-Application-Project/wiki/Optional-Task-9:-Efficiency)

CODE
------

The runtimes of the various trials are found in the folder timing_data

The file plot_times.py generates the plots in the graphs directory using the timing data. It also performs t-tests to determine if any of the numtimes are significantly faster.

Libraries used include numpy, scikit-learn, and matplotlib



