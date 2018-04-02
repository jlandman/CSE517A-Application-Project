from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def convertLabel(y):
    if y < median:
        return 0
    else:
        return 1

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

N = X.shape[0]
X_use = X[0:N, :]
Y_use = Y[0:N, :]
X_hold = X[N:, :]
Y_hold = Y[N:, :]

print 'Standardizing data...'
scaler = StandardScaler()
scaler = scaler.fit(X_use)
X_use = scaler.transform(X_use)

print 'Performing PCA...'
pca = PCA(n_components=2)
pca.fit(X_use)
X_use = pca.transform(X_use)

print 'Converting truth labels...'
median = np.median(Y_use)
labelConvertVectorized = np.vectorize(convertLabel)
Y_use = labelConvertVectorized(Y_use)

classZero = np.where(Y_use == 0)
classOne = np.where(Y_use == 1)
X_zero = X_use[classZero, :]
X_one = X_use[classOne, :]

print 'Plotting data...'
fig = plt.figure()

plt.scatter(X_zero[:, 0], X_zero[:, 1], c='r', marker='o', label='0')
plt.scatter(X_one[:, 0], X_one[:, 1], c='c', marker='^', label='1')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title(r'Plot of Data Points in X Colored by Y (PCA)')
plt.legend()

# ax = fig.gca(projection='3d')
# ax.scatter(X_zero[:, 0], X_zero[:, 1], X_zero[:, 2], c='r', marker='o', label='0')
# ax.scatter(X_one[:, 0], X_one[:, 1], X_one[:, 2], c='b', marker='^', label='1')
# ax.set_xlabel(r'PC 1')
# ax.set_ylabel(r'PC 2')
# ax.set_zlabel(r'PC 3')
# ax.set_title(r'Plot of Data Points in X Colored by Y (PCA)')
# ax.legend()

plt.show()

# alphas = 10.0**np.arange(-6, 5, .05)
# bestAlpha = 0
# bestVal = 1
# valAccs = np.empty((alphas.shape))
# print 'Cross-validating for best penalty hyperparameter...'
# for alpha in xrange(len(alphas)):
#     print '\tTraining logistic regression classifier with alpha={}'.format(alphas[alpha])
#     vals = []

#     for t in xrange(10):
#         print '\t\tIteration {}'.format(t+1)
#         X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
#         clf = SGDClassifier(loss='log', penalty='l2', alpha=alphas[alpha], learning_rate='optimal')
#         clf.fit(X_train, y_train)
#         val = 1.0 - clf.score(X_test, y_test)
#         vals.append(val)

#     valAccs[alpha] = np.mean(np.array(vals))

#     if np.mean(np.array(vals)) < bestVal:
#         bestVal = np.mean(np.array(vals))
#         bestAlpha = alphas[alpha]

# np.savetxt('pca_lr_vals.txt', valAccs)
# print 'Best alpha = {}'.format(bestAlpha)
# print 'Best validation error = {}'.format(bestVal)
# print repr(valAccs)
