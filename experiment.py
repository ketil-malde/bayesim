import bayesim as b
import numpy as np
from math import sqrt

# run a network, generate plots by iterating it
#  a) prob distr of outputs
#  b) correlation matrix

def plot_net(network):
    return None

# ref https://en.wikipedia.org/wiki/Correlation_and_dependence
def correlate(data):
    # n samples (rows), m variables (columns)
    (n,m) = data.shape

    # sum up x and x² for all elements
    sx = np.sum(data, axis=0) # sum elements in all rows
    sx2 = np.sum(data*data, axis=0) # sum square of elements (is the same for booleans!)

    # Count co-occurrences
    res = np.zeros(shape=(m,m))
    for x in data:
        res = res + np.outer(x,x) # huh, this is only when both are one?

    # Calculate correlations r_{i,j}
    for i in range(m):
        stdev_i = sqrt(n*sx2[i]-sx[i]**2)
        for j in range(m):
            stdev_j = sqrt(n*sx2[j]-sx[j]**2)
            if stdev_i != 0 and stdev_j != 0:
                res[i,j] = (n*res[i,j]-sx[i]*sx[j]) / (stdev_i*stdev_j)
            else:
                res[i,j] = 0
    return res

# Testing:    
# correlate(np.array([[1,0],[0,1],[1,1]]))

import matplotlib.pyplot as plt
import seaborn as sns

x = np.array(b.iterate(b.mklayerednet(10,[(10,2),(10,3),(10,3)])))
a = correlate(x)
sns.heatmap(a[:10], cmap='coolwarm', linewidth=0.5)
# plt.xtics([9.5,19.5]) how do I do this?
plt.grid(True)
plt.show()

# for n networks of given complexity
#   run network x times to generate data
#   learn output using {SVM, RF, KNN, ANN} (how to optimize parameters?)
#   test accuracy on iteration over all data

classifiers = ['SVM', 'RF', 'KNN', 'ANN']

def evalnet(network, n, x):
    for i in range(n):
        simdata = b.simulate(network, iterations=x)
        for c in classifiers:
            # learn the classifiers on data
            cls = learn(c, simdata)
            testdata = iterate(network)
            test(cls, testdata)
            # test the classifier and output
    return None

from sklearn import svm

def learn(cls, data):
    if cls=='SVM':
        # split data into input x and output labels y
        c = svm.SVC(gamma='scale')
        return c.fit(datax, datay)
    else:
        raise Exception(f'No such classifier: {cls}')

def test(cls, data):
    return None


