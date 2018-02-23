from sklearn import linear_model
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score

with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:
    input = file.readlines()

names = input[0]
names = names.split(',')[2:]
data = input[1:]
data = [line.split(',') for line in data]
data = np.array(data)
np.random.shuffle(data)
X = data[:, 2:-1].astype(float)
Y = data[:, -1:].astype(float)
Y = np.log(Y)

numPoints = len(Y)
pTrain = 0.8
index = int(numPoints*pTrain)
Xtr = X[index:]
Ytr = Y[index:]
Xte = X[:index]
Yte = Y[:index]

reg = linear_model.Ridge(alpha=0.05, fit_intercept=True, normalize=True)
reg.fit(Xtr, Ytr)
coeffs = reg.coef_.flatten()
abs_coeffs = np.absolute(coeffs)

abs_coeffs_copy = np.copy(abs_coeffs)
for i in range(1,6):
    maxindex = np.argmax(abs_coeffs_copy)
    print('Rank: ',i,' name: ',names[maxindex],' value: ',coeffs[maxindex])
    abs_coeffs_copy[maxindex] = 0

featureSplit = np.split(abs_coeffs, [11,17,26,29,37,42])
labels = ('numItems', 'article_class', 'kw', 'self_ref', 'dayWeek', 'LDA', 'textStat')
pyplot.boxplot(featureSplit, labels=labels)
pyplot.title('Magnitude Feature Weights')
pyplot.draw()
pyplot.show()



















