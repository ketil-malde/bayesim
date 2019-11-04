import bayesim as b
import numpy as np

# for n networks of given complexity
#   run network x times to generate data
#   learn output using {SVM, RF, KNN, ANN} (how to optimize parameters?)
#   test accuracy on iteration over all data
# maybe also test with layer [1] as input (non-uniform distribution)?
classifiers = ['SVM', 'KNN'] # , 'RF', 'KNN', 'ANN']

def evalnet(network, times, iters):
    res = {}
    for c in classifiers:
        res[c]=[]
    for i in range(times):
        simdata = b.simulate(network, iterations=iters)
        # x are inputs, y is the last node output
        xs, ys = simdata[:, :network.input_size], simdata[:, -1]
        testdata = b.simulate(network, iterations=1000)
        test_xs, test_ys = testdata[:, :network.input_size], testdata[:, -1]
        for c in classifiers:
            # learn the classifiers on data
            cls = learn(c, xs, ys)
            acc = test(cls, test_xs, test_ys)
            res[c].append(acc)
    return res

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def learn(cls, xs, ys):
    if cls=='SVM':
        c = svm.SVC(gamma='scale')
    elif cls == 'KNN':
        c = KNeighborsClassifier()
    else:
        raise Exception(f'No such classifier: {cls}')
    return c.fit(xs, ys)

# return accuracy of classifier    
def test(cls, xs, ys):
    return sum(cls.predict(xs) == ys)/len(xs)

r = evalnet(b.mklayerednet(10, [(10,2),(5,3),(1,5)]), 10, 100)
print(r)

