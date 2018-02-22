import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt



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

    alph = 0.8
    n = len(X)
    ntr = int(alph*n)
    nte = n-ntr

    Xtr = X[:ntr,:]
    Ytr = Y[:ntr,:]
    Xte = X[ntr:,:]
    Yte = Y[ntr:,:]

    
