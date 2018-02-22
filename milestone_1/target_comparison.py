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
    
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(35, 10))
    fig.set_size_inches(15.0, 9.0, forward=True)
    #fig.set_tight_layout(True)
    fig.suptitle('Target Variable Distribution',fontsize=16)
    fig.set_dpi(500)
    ax0, ax1 = axes    

    ax0.hist(Y, normed=False, bins=10000)
    ax0.set_ylabel('Number of Articles');
    ax0.set_xlabel('Shares');
    #plt.draw()
    #plt.savefig('share_histogram_log.png')

    ax1.hist(Ylog, normed=False, bins=30)
    ax1.set_ylabel('Number of Articles');
    ax1.set_xlabel('ln(Shares)');
    #plt.ylim((0,1000))


    #fig = plt.gcf()


    #plt.draw()
    plt.savefig('target_comparison.png')



