from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
import numpy as np

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

print 'Converting truth labels...'
median = np.median(Y_use)

labelConvertVectorized = np.vectorize(convertLabel)
Y_use = labelConvertVectorized(Y_use)

pca_alpha = 0.0177827941004
lr_alpha = 0.00501187233627

pca2_results = []
pca3_results = []
lr_results = []

numCV = 50
for i in xrange(numCV):
    pca2Vals = []
    pca3Vals = []
    lrVals = []
    print '\tCV {}/{}'.format(i+1, numCV)
    
    for j in xrange(10):
        pca2 = PCA(n_components=2)
        X_pca2 = pca2.fit_transform(X_use)
        pca3 = PCA(n_components=3)
        X_pca3 = pca3.fit_transform(X_use)

        print '\t\tIteration {}'.format(j+1)

        X_train, X_test, y_train, y_test = train_test_split(X_pca2, Y_use.reshape((Y_use.shape[0],)), test_size=0.1)
        pca2Clf = SGDRegressor(loss='log', penalty='l2', alpha=pca_alpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
        pca2Clf.fit(X_train, y_train)
        pca2Val = pca2Clf.score(X_test, y_test)
        pca2Vals.append(pca2Val)

        X_train = pca3.transform(pca2.inverse_transform(X_train))
        X_test = pca3.transform(pca2.inverse_transform(X_test))
        pca3Clf = SGDRegressor(loss='log', penalty='l2', alpha=pca_alpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
        pca3Clf.fit(X_train, y_train)
        pca3Val = pca3Clf.score(X_test, y_test)
        pca3Vals.append(pca3Val)

        X_train = pca3.inverse_transform(X_train)
        X_test = pca3.inverse_transform(X_test)
        lrClf = SGDRegressor(loss='log', penalty='l2', alpha=lr_alpha, learning_rate='optimal', max_iter=1000, tol=1e-3)
        lrClf.fit(X_train, y_train)
        lrVal = lrClf.score(X_test, y_test)
        lrVals.append(lrVal)

    pca2_results.append(np.mean(np.array(pca2Vals)))
    pca3_results.append(np.mean(np.array(pca3Vals)))
    lr_results.append(np.mean(np.array(lrVals)))

np.savetxt('pca2_cv_scores.txt', pca2_results)
np.savetxt('pca3_cv_scores.txt', pca3_results)
np.savetxt('lr_cv_scores.txt', lr_results)

print 'Means:'
print 'PCA (2 components):\t{}'.format(np.mean(pca2_results))
print 'PCA (3 components):\t{}'.format(np.mean(pca3_results))
print 'No PCA:\t{}'.format(np.mean(lr_results))
print 

fResults = f_oneway(pca2_results, pca3_results, lr_results)
print 'ANOVA Results:'
print 'F = {}'.format(fResults.statistic)
print 'p = {}'.format(fResults.pvalue)
print

pca2LrResults = ttest_rel(pca2_results, lr_results)
print 'T-test Results: PCA (2-components) vs. No PCA'
print 't = {}'.format(pca2LrResults.statistic)
print 'p = {}'.format(pca2LrResults.pvalue)
print

pca3LrResults = ttest_rel(pca3_results, lr_results)
print 'T-test Results: PCA (3-components) vs. No PCA'
print 't = {}'.format(pca3LrResults.statistic)
print 'p = {}'.format(pca3LrResults.pvalue)
print

pca2pca3Results = ttest_rel(pca2_results, pca3_results)
print 'T-test Results: PCA (2-components) vs. PCA (3-components)'
print 't = {}'.format(pca2pca3Results.statistic)
print 'p = {}'.format(pca2pca3Results.pvalue)
print
