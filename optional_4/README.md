<h2>Optional Task 4: SVM</h2>

This optional task dealt with the creation of a support vector machine for classification purposes. The SVM used had a RBF kernel. Using 10-fold cross validation, hyperparameters __C__ and __gamma__ were chosen such that they minimized the validation error of the SVM. The data was split into classes based on a median split by the number of shares, our target feature.

To train the SVM, just run `python classifysvm.py` on a textfile containing our data. The output of the script are the optimal hyperparameters for the SVM.

The following is a surface plot showing the validation error surface of the SVM:
![SVM Error Plot](svm_err_graph.png)
