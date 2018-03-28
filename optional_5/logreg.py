from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import numpy as np

with open('data.txt', 'r') as f:
    names = f.readline().split()
    data = np.loadtxt(f)
    X = data[:, 2:-1]
    Y = data[:, -1]

shuffle = np.arange(X.shape[0])
shuffle = np.random.shuffle(shuffle)
X = X[shuffle, :]
X = X.reshape((X.shape[1], X.shape[2]))
Y = Y[:, shuffle]

N = 4000
X_use = X[0:N, :]
Y_use = Y[0:N, :]
X_hold = X[N:, :]
Y_hold = Y[N:, :]

print 'Standardizing data...'
scaler = StandardScaler()
scaler = scaler.fit(X_use)
X_use = scaler.transform(X_use)

print 'Converting truth labels...'
median = np.median(Y_use)

def convertLabel(y):
    if y < median:
        return 0
    else:
        return 1

labelConvertVectorized = np.vectorize(convertLabel)
Y_use = labelConvertVectorized(Y_use)

alphas = 10.0**np.arange(-6, 5, 0.05)

bestAlpha = 0
bestVal = 1
valAccs = np.empty((alphas.shape))

print 'Cross-validating for best penalty hyperparameter...'
for alpha in xrange(len(alphas)):
    print '\tTraining logistic regression classifier with alpha={}'.format(alphas[alpha])
    vals = []

    for t in xrange(10):
        print '\t\tIteration {}'.format(t+1)
        X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
        clf = SGDClassifier(loss='log', penalty='l2', alpha=alphas[alpha], learning_rate='optimal')
        clf.fit(X_train, y_train)
        val = 1.0 - clf.score(X_test, y_test)
        vals.append(val)

    valAccs[alpha] = np.mean(np.array(vals))

    if np.mean(np.array(vals)) < bestVal:
        bestVal = np.mean(np.array(vals))
        bestAlpha = alphas[alpha]

np.savetxt('lr_vals.txt', valAccs)
print 'Best alpha = {}'.format(bestAlpha)
print 'Best validation error = {}'.format(bestVal)
print repr(valAccs)