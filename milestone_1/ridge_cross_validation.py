import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def cross_val_error(X,Y,lam):

    insample = 0.0
    outsample = 0.0

    kf = KFold(10)
    for train_index, test_index in kf.split(X):

        Xtr = X[train_index]
        Ytr = Y[train_index]
        Xte = X[test_index]
        Yte = Y[test_index]

        reg_ridge = linear_model.Ridge(alpha=lam,fit_intercept=True, normalize=True)
        reg_ridge.fit(Xtr,Ytr)

        Ytr_pred = reg_ridge.predict(Xtr)
        Yte_pred = reg_ridge.predict(Xte)
        
        resids_tr = (Ytr-Ytr_pred).flatten()
        resids_te = (Yte-Yte_pred).flatten()

        insample += np.sqrt(np.sum(np.dot(resids_tr,resids_tr))/Ytr.size)
        outsample += np.sqrt(np.sum(np.dot(resids_te,resids_te))/Yte.size)

    insample /= 10.0
    outsample /= 10.0

    return (insample,outsample)


if __name__ == "__main__":

    with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:
        input = file.readlines()

    names = input[0]
    data = input[1:]

    data = [line.split(',') for line in data]
    data = np.array(data)
    
    np.random.shuffle(data)

    X = data[:,2:-1].astype(float)
    Y = data[:,-1:].astype(float)
    Y = np.log(Y)

    print('Running 10-Fold Cross-Validation, Ridge Regression Model')

    lnlambdas = np.arange(-10,10,0.1)
    lambdas = np.exp(lnlambdas)
    insample = np.zeros_like(lambdas)
    outsample = np.zeros_like(lambdas)
    for i, l in enumerate(lambdas):
        insample[i], outsample[i] = cross_val_error(X,Y,l)
        #print('lambda: %f | in sample error: %.5f | out of sample error: %.5f' % (lambdas[i], insample[i], outsample[i]))
    
    min_index = np.argmin(outsample)
    print('minimum lambda: %f | in sample error: %.5f | out of sample error: %.5f' % (lambdas[min_index], insample[min_index], outsample[min_index]))

    
    plt.plot(lnlambdas,insample, label='in sample error')
    plt.plot(lnlambdas,outsample, label='out of sample error')
    plt.ylabel('Error');
    plt.xlabel('ln(lambda)');
    plt.legend()

    plt.draw()
    plt.savefig('ridge_lambda.png')
    print('saved to \'ridge_lambda.png\'')
