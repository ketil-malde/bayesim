# using a random bayesian network to generate data parametrized by complexity

import numpy as np

# A node is an array of k indices, and a 2^k array of probabilities
# 'width' is the number of indices, and 'entropy' the distribution of probabilities
def mknode(inputwidth, width, entropy=1):
    ix = np.random.randint(inputwidth, size=width)
    # todo: use entropy, maybe: select gaussian g => g if g>0, 1-g if g<0, min 0, max 1 
    ps = np.random.beta(entropy, entropy, size=2**width)
    return (ix,ps)

# returns the node probability based on input
def apply_node_prob(indata, node):
    # todo: verify indata has correct size for node
    (ix,ps) = node
    p = indata[ix]
    x = sum(2**i for i, v in enumerate(reversed(p)) if v) # interpret inp as a base2 index into ps
    return ps[x]

def get_node_outputs(node_probs):
    return np.random.binomial(1,node_probs)

# A network consists of layers
# maybe: 'layers' is a list of layer sizes and widths
def mknet(input_size, layers):
    # need to remember the input size?
    return None

# Simulate data with a generator
def sim_gen(network):
    return None

# Simulate a bunch of data    
def simulate(network, count):
    return None

# Testing
# n = mknode(4,2,1)
# apply_node_prob(np.array([0,1,0,1], n)
