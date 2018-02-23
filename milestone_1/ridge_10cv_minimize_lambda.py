import numpy as np
import math
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
    data = data[:,2:].astype(float)

    lamb = 1.0
    lamb_old = lamb*1.1
    temps = np.arange(0.0,10.0,.01)
    lambs = np.zeros_like(temps)

    for i in range(0,len(temps)):

        T = temps[i]
        np.random.shuffle(data)
        X = data[:,:-1]
        Y = data[:,-1:]

        inerr_old, outerr_old = cross_val_error(X,Y,lamb_old)
        inerr_new, outerr_new = cross_val_error(X,Y,lamb)
        
        activation = 1.0/(1.0+np.exp(-T))
        print("error old: %.5f || error new: %.5f || diff: %.5f" % (outerr_old, outerr_new, outerr_new-outerr_old))
        print("lamda old: %f || lamda new: %f " %(lamb,lamb_old))
        print("activation: %f" %(activation))

        if outerr_new < outerr_old:
            lamb_old, lamb = lamb, lamb*lamb/lamb_old
            print('found a lower error')
        elif np.random.ranf(1)[0] > activation:
            lamb_old, lamb = lamb, lamb*lamb/lamb_old
            print('random move')
        else:
            lamb_old, lamb = lamb, lamb_old
            print('reversed')

        lambs[i]=lamb
    
    lambda_avg = np.average(lambs[(int(len(temps)/2)):])  
    print('average lambda: %.5f: ' % (lambda_avg))
    plt.plot(lambs, 'bo-', label='lambda')
    plt.suptitle('Lambda Minimization',fontsize=16)
    plt.ylabel('Lambda');
    plt.xlabel('Iteration');

    plt.savefig('min_lambda.png',dpi=500)
    print('saved to \'min_lambda.png\'')
