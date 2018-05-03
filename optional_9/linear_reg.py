from sklearn import linear_model
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from time import time

with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:
    input = file.readlines()

names = input[0]
names = names.split(',')[2:]
data = input[1:]
data = [line.split(',') for line in data]
data = np.array(data)
data = data[:,2:].astype(float)
data[:,-1:]=np.log(data[:,-1:])


numIters = 100
trainTimeData = [0] * numIters
testTimeData = [0] * numIters
trainError = []
testError = []


for i in range(0,numIters):
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1:]

    numPoints = len(Y)
    pTrain = 0.9
    index = int(numPoints*pTrain)
    Xtr = X[index:,:]
    Ytr = Y[index:,:]
    Xte = X[:index,:]
    Yte = Y[:index,:]

    #Start Timing (data split is not part of timing)
    startTrain = time()
    reg = linear_model.Ridge(alpha=0.05, fit_intercept=True, normalize=True)
    reg.fit(Xtr, Ytr)
    startTest = time()
    Ytr_pred = reg.predict(Xtr)
    Yte_pred = reg.predict(Xte)
    end = time()
    trainError.append(mean_absolute_error(Ytr, Ytr_pred))
    testError.append(mean_absolute_error(Yte, Yte_pred))

    trainTimeData[i] = startTest - startTrain
    testTimeData[i] = end - startTest


print("NumSamples: ",numIters)
print("Mean Training Time: ",np.mean(trainTimeData))
print("Mean Testing Time: ",np.mean(testTimeData))
print("Mean Train Error: ",np.mean(trainError))
print("Mean Test Error: ",np.mean(testError))

np.savetxt('timing_data/train_time_ridge.csv', trainTimeData)
np.savetxt('timing_data/test_time_ridge.csv', testTimeData)
# np.savetxt('../milestone_4/error_data/train_error_ridge.csv', trainError)
# np.savetxt('../milestone_4/error_data/test_error_ridge.csv', testError)





















