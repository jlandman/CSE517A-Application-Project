from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
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
bestC = 1
bestG = 0.00390625
bestAlpha = 0.0446683592151

svm_results = []
lr_results = []

numCV = 50
for i in xrange(numCV):
    svmVals = []
    lrVals = []
    print '\tCV {}/{}'.format(i+1, numCV)
    
    for j in xrange(10):
        X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
        print '\t\tIteration {}'.format(j+1)
        svmClf = svm.SVC(kernel=kernel, C=bestC, gamma=bestG)
        svmClf.fit(X_train, y_train)
        svmVal = svmClf.score(X_test, y_test)
        svmVals.append(svmVal)

        lrClf = SGDClassifier(loss='log', penalty='l2', alpha=bestAlpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
        lrClf.fit(X_train, y_train)
        lrVal = lrClf.score(X_test, y_test)
        lrVals.append(lrVal)

    svm_results.append(np.mean(np.array(svmVals)))
    lr_results.append(np.mean(np.array(lrVals)))

np.savetxt('svm_cv_scores.txt', svm_results)
np.savetxt('lr_cv_scores.txt', lr_results)

tResults = ttest_rel(svm_results, lr_results)
print 't = {}'.format(tResults.statistic)
print 'p = {}'.format(tResults.pvalue)