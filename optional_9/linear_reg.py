from sklearn import linear_model
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

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
inMean = np.zeros(numIters)
inMeanSq = np.zeros(numIters)
inR_2 = np.zeros(numIters)
outMean = np.zeros(numIters)
outMeanSq = np.zeros(numIters)
outR_2 = np.zeros(numIters)
inExpVar = np.zeros(numIters)
outExpVar = np.zeros(numIters)
for i in range(0,numIters):
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1:]

    numPoints = len(Y)
    pTrain = 0.8
    index = int(numPoints*pTrain)
    Xtr = X[index:,:]
    Ytr = Y[index:,:]
    Xte = X[:index,:]
    Yte = Y[:index,:]

    reg = linear_model.Ridge(alpha=0.05, fit_intercept=True, normalize=True)
    reg.fit(Xtr, Ytr)

    Ytr_pred = reg.predict(Xtr)
    Yte_pred = reg.predict(Xte)

    inMean[i] = mean_absolute_error(Ytr,Ytr_pred)
    inMeanSq[i] = mean_squared_error(Ytr, Ytr_pred)
    inR_2[i] = r2_score(Ytr, Ytr_pred)
    inExpVar[i] = explained_variance_score(Ytr, Ytr_pred)
    outMean[i] = mean_absolute_error(Yte,Yte_pred)
    outMeanSq[i] = mean_squared_error(Yte, Yte_pred)
    outR_2[i] = r2_score(Yte, Yte_pred)
    outExpVar[i] = explained_variance_score(Yte, Yte_pred)

print("In Sample:")
print("Mean Error: ",np.mean(inMean))
print("Mean Squared Error: ",np.mean(inMeanSq))
print("R^2 score: ",np.mean(inR_2))

print("Out of Sample:")
print("Mean Error: ",np.mean(outMean))
print("Mean Squared Error: ",np.mean(outMeanSq))
print("R^2 score: ",np.mean(outR_2))



















