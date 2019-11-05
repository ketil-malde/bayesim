import bayesim as b
import numpy as np

# for n networks of given complexity
#   run network x times to generate data
#   learn output using {SVM, RF, KNN, ANN} (how to optimize parameters?)
#   test accuracy on iteration over all data
# maybe also test with layer [1] as input (non-uniform distribution)?
classifiers = ['SVM', 'KNN', 'RF', 'ANN']

def evalnet(network, times, iters):
    res = {}
    for c in classifiers:
        res[c]=[]
    for i in range(times):
        simdata = b.simulate(network, iterations=iters)
        # x are inputs, y is the last node output
        xs, ys = simdata[:, :network.input_size], simdata[:, -1]
        if len(np.unique(ys)) < 2: continue
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def learn(cls, xs, ys):
    if cls=='SVM':
        c = svm.SVC(gamma='scale')
    elif cls == 'KNN':
        c = KNeighborsClassifier()
    elif cls == 'RF':
        c = RandomForestClassifier(n_estimators=10)
    elif cls == 'ANN':
        c = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 4), random_state=1)
    else:
        raise Exception(f'No such classifier: {cls}')
    return c.fit(xs, ys)

# return accuracy of classifier
def test(cls, xs, ys):
    return sum(cls.predict(xs) == ys)/len(xs)

# r = evalnet(b.mklayerednet(10, [(10,2),(5,3),(1,5)]), 10, 100)
# print(r)

# Run experiments

from random import randint, choice

# Generate a config as a dictionary
def gen_config():
    conf = {}
    conf['layer_depth'] = randint(3, 12)
    conf['layer_width'] = randint(6, 40)
    conf['node_width']  = randint(2, 5)
    conf['entropy']     = choice([0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0])
    conf['data_size']   = round(10**(randint(8, 16)/4))  # 100..10K data points
    return conf

def build_network(cf):
    insize = cf['layer_width']
    layers = []
    w = insize
    for i in range(cf['layer_depth']-1):
        factor = 1-i/cf['layer_depth']
        layers.append((round(insize*factor), cf['node_width']))
    layers.append((1, cf['node_width']))
    print(layers)
    return b.mklayerednet(insize, layerlist=layers, entropy=cf['entropy'])

def print_conf_res(conf, res):
    print(conf['layer_depth'], conf['layer_width'], conf['node_width'], conf['entropy'], conf['data_size'], res)

def run(x):
    for _ in range(x):
        cf = gen_config()
        print(cf)
        net = build_network(cf)
        res = evalnet(net, 10, cf['data_size'])
        print_conf_res(cf, res)

# cf = gen_config()
# print(cf)
# net = build_network(cf)
# print(net)
run(10)
