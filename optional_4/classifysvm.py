from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
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

kernel = 'rbf'
C = np.array([2**i for i in xrange(-3, 3)])
gamma = np.array([2**i for i in xrange(-8,-1)])
k = 10

# This commented-out block calculates the best C and gamma values.
# It was commented out due to computational cost after the first time best C and best gamma were calculated.
bestC = 0
bestG = 0
bestVal = 1
valErrs = np.empty((len(C), len(gamma)))

print 'Cross-validating for best penalty and gamma hyperparameters...'
for c in xrange(len(C)):
    for g in xrange(len(gamma)):
        print '\tTraining SVM with {} kernel, C={}, and gamma={}'.format(kernel, C[c], gamma[g])
        vals = []

        for t in xrange(10):
            print '\t\tIteration {}'.format(t+1)
            X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
            clf = svm.SVC(kernel=kernel, C=C[c], gamma=gamma[g])
            clf.fit(X_train, y_train)
            val = 1.0 - clf.score(X_test, y_test)
            vals.append(val)

        valErrs[c, g] = np.mean(np.array(vals))

        if np.mean(np.array(vals)) < bestVal:
            bestVal = np.mean(np.array(vals))
            bestC = C[c]
            bestG = gamma[g]

np.savetxt('svm_vals.txt', valErrs)
print 'Best C = {}, best gamma = {}'.format(bestC, bestG)
print 'Best validation error = {}'.format(bestVal)

print valErrs