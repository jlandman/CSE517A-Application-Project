from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
import numpy as np
from time import time
from sklearn.metrics import mean_absolute_error

with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:
    input = file.readlines()

names = input[0]
names = names.split(',')[2:]
data = input[1:]
data = [line.split(',') for line in data]
data = np.array(data)
data = data[:,2:].astype(float)
data[:,-1:]=np.log(data[:,-1:])

pca_alpha = 0.0177827941004
lr_alpha = 0.00501187233627

pca2_trainError = []
pca2_testError = []
pca2_trainTime = []
pca2_testTime = []

pca3_trainError = []
pca3_testError = []
pca3_trainTime = []
pca3_testTime = []


numIters = 100
for i in xrange(numIters):
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1:]

    N = X.shape[0]
    X_use = X[0:N, :]
    Y_use = Y[0:N, :]
    X_hold = X[N:, :]
    Y_hold = Y[N:, :]

    scaler = StandardScaler()
    scaler = scaler.fit(X_use)
    X_use = scaler.transform(X_use)

    median = np.median(Y_use)


    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_use)
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_use)

    X_train, X_test, y_train, y_test = train_test_split(X_pca2, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
    pca2Clf = SGDRegressor(penalty='l2', alpha=pca_alpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
    startTrain = time()
    pca2Clf.fit(X_train, y_train)
    startTest = time()
    Ytr_pred = pca2Clf.predict(X_train)
    Yte_pred = pca2Clf.predict(X_test)
    end = time()
    pca2_trainError.append(mean_absolute_error(y_train, Ytr_pred))
    pca2_testError.append(mean_absolute_error(y_test, Yte_pred))
    pca2_trainTime.append(startTest - startTrain)
    pca2_testTime.append(end - startTest)

    X_train, X_test, y_train, y_test = train_test_split(X_pca3, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
    pca3Clf = SGDRegressor(penalty='l2', alpha=pca_alpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
    startTrain = time()
    pca3Clf.fit(X_train, y_train)
    startTest = time()
    Ytr_pred = pca3Clf.predict(X_train)
    Yte_pred = pca3Clf.predict(X_test)
    end = time()
    pca3_trainError.append(mean_absolute_error(y_train, Ytr_pred))
    pca3_testError.append(mean_absolute_error(y_test, Yte_pred))
    pca3_trainTime.append(startTest - startTrain)
    pca3_testTime.append(end - startTest)

print("NumSamples: ",numIters)
print("Average Mean Absolute Training Error:")
print("\tPCA2: ",np.mean(pca2_trainError))
print("\tPCA3: ",np.mean(pca3_trainError))

print
print("Average Mean Absolute Testing Error:")
print("\tPCA2: ",np.mean(pca2_testError))
print("\tPCA3: ",np.mean(pca3_testError))

print
print("Average Training Time:")
print("\tPCA2: ",np.mean(pca2_trainTime))
print("\tPCA3: ",np.mean(pca3_trainTime))

print
print("Average Testing Time:")
print("\tPCA2: ",np.mean(pca2_testTime))
print("\tPCA3: ",np.mean(pca3_testTime))

np.savetxt('timing_data/train_time_pca2.csv', pca2_trainTime)
np.savetxt('timing_data/test_time_pca2.csv', pca2_testTime)
# np.savetxt('../milestone_4/error_data/train_error_pca2.csv', pca2_trainError)
# np.savetxt('../milestone_4/error_data/test_error_pca2.csv', pca2_testError)

np.savetxt('timing_data/train_time_pca3.csv', pca3_trainTime)
np.savetxt('timing_data/test_time_pca3.csv', pca3_testTime)
# np.savetxt('../milestone_4/error_data/train_error_pca3.csv', pca3_trainError)
# np.savetxt('../milestone_4/error_data/test_error_pca3.csv', pca3_testError)