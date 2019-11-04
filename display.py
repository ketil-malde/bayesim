# Testing:
# use: python3 experiment.py | dot -Tpdf -onet.pdf

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

import bayesim as b

# run a network, generate plots by iterating it
#  a) prob distr of outputs
#  b) correlation matrix

def plot_net(network):
    return None

# ref https://en.wikipedia.org/wiki/Correlation_and_dependence
def correlate(data):
    # n samples (rows), m variables (columns)
    (n, m) = data.shape

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

# Generate network and data
inputs = 10
layerszs = [10, 10, 5, 1]
widths = [2, 3, 3, 5]
net = b.mklayerednet(inputs, zip(layerszs, widths))
x = np.array(b.iterate(net))
a = correlate(x)

# Print net to dot files
b.net_to_dot(net)
sys.stdout.flush()


# Calculate actual effect of an input
# i.e. for all inputs, how many cases does flipping input i change final outcome
# Print (to stderr) the fraction of cases where input i flips the outcome
for i in range(inputs):
    x0i = x[:, i] == 0
    x0 = x[x0i]
    x1i = x[:, i] == 1
    x1 = x[x1i]
    z = x0[:, -1] == x1[:, -1]
    print(a[i, -1], 1-sum(z)/2**(inputs-1), file=sys.stderr)

# plot output
fig, axn = plt.subplots(1, len(layerszs)+1, sharex=False, sharey='row')
sns.heatmap(a[:inputs,:inputs], ax=axn.flat[0], cmap='coolwarm', linewidth=0.5, vmin=-1, vmax=1, cbar=False)
tmp = inputs
for i, w in enumerate(layerszs):
    sns.heatmap(a[:inputs, tmp:tmp+w], ax=axn.flat[i+1], cmap='coolwarm', linewidth=0.5,
                vmin=-1, vmax=1, cbar=True if i+1==len(layerszs) else False, annot=True if i+1==len(layerszs) else False)
    tmp=tmp+w
fig.tight_layout()
plt.show()
