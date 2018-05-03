from sklearn.model_selection import train_test_split
from sklearn import svm
from time import time
from sklearn.metrics import mean_absolute_error
import numpy as np

with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:
    input = file.readlines()

names = input[0]
names = names.split(',')[2:]
data = input[1:]
data = [line.split(',') for line in data]
data = np.array(data)
data = data[:,2:].astype(float)
data[:,-1:]=np.log(data[:,-1:])

numIters = 10
# numTrainPts = 5000
# numTestPts = 3979

trainTime = []
testTime =[]
trainError = []
testError = []

for i in range(numIters):
    print("Iteration {}".format(i))
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1:]
    # X_train = data[:numTrainPts, :-1]
    # y_train = data[:numTrainPts, -1:]
    # X_test = data[numTrainPts:(numTestPts+numTrainPts), :-1]
    # y_test = data[numTrainPts:(numTestPts+numTrainPts), -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y.reshape((Y.shape[0],)), test_size=0.1)
    clf = svm.SVR(kernel='rbf', C=2.0, gamma=0.0078125)
    startTrain = time()
    clf.fit(X_train, y_train)
    startTest = time()
    Ytr_pred = clf.predict(X_train)
    Yte_pred = clf.predict(X_test)
    end = time()
    trainError.append(mean_absolute_error(y_train, Ytr_pred))
    testError.append(mean_absolute_error(y_test, Yte_pred))
    trainTime.append(startTest - startTrain)
    testTime.append(end - startTest)

print("NumSamples: ",numIters)
print("Mean Training Time: ",np.mean(trainTime))
print("Mean Testing Time: ",np.mean(testTime))
print("Mean Train Error: ",np.mean(trainError))
print("Mean Test Error: ",np.mean(testError))

np.savetxt('timing_data/train_time_svm.csv', trainTime)
np.savetxt('timing_data/test_time_svm.csv', testTime)
np.savetxt('../milestone_4/error_data/train_error_svm.csv', trainError)
np.savetxt('../milestone_4/error_data/test_error_svm.csv', testError)
