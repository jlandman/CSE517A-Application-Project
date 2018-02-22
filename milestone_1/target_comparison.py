import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

with open('../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv', 'r') as file:

    input = file.readlines()
    names = input[0]
    data = input[1:]

    data = [line.split(',') for line in data]
    data = np.array(data)

    np.random.shuffle(data)

    X = data[:,2:-1].astype(float)
    Y = data[:,-1:].astype(float)
    Ylog = np.log(Y)

    # Generate graphs showing the distribution of Y and ln(Y)

    plt.hist(Ylog, normed=False, bins=30)
    plt.ylabel('Number of Articles');
    plt.xlabel('ln(Shares)');
    plt.draw()
    plt.savefig('share_histogram_log.png')

    plt.hist(Y, normed=False, bins=10000)
    plt.ylabel('Number of Articles');
    plt.xlabel('Shares');
    plt.ylim((0,1000))
    plt.draw()
    plt.savefig('share_histogram_raw.png')



