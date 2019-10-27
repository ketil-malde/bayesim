# using a random bayesian network to generate data parametrized by complexity

import numpy as np

# A node is a tuple of an array of w indices, and a 2^w array of probabilities
# 'width' is w, i.e. the number of inputs to each node
# 'entropy' is the alpha/beta parameter used to draw probabilities from a beta distribution (ie., not to be confused with actual entropy)
# An 'entropy' of zero means a deterministic network, where the same input always give the same outputs in each node.
def mknode(inputwidth, width, first=0, entropy=0):
    ix = first+np.random.randint(inputwidth, size=width)
    if entropy == 0:
        ps = np.random.choice([0,1], size=2**width)
    else:
        ps = np.random.beta(entropy, entropy, size=2**width)
    return (ix, ps)

# Look up the node probability based on values in the inputs
def apply_node(indata, node):
    # todo: verify indata has correct size for node
    (inp_idx, ps) = node
    inp = indata[inp_idx]
    x = sum(2**i for i, v in enumerate(reversed(inp)) if v) # interpret inp as a base2 index into ps
    return ps[x]

# Create a network given input size, number of (non-input nodes), the input width, and entropy.
# The returned network is a tuple consisting of the input size and the list of nodes.
def mknet(input_size, output_size, width, entropy=0):
    net = []
    for i in range(input_size, input_size+output_size):
        net.append(mknode(i, width, entropy))
    return (input_size, net)

# ignore entropy for now
def mklayerednet(input_size, layerlist, entropy=0):
    net = []
    prev_size = input_size
    prev_start = 0
    for size, width in layerlist:
        for n in range(size):
            net.append(mknode(prev_size, width, first=prev_start, entropy=entropy))
        prev_size = size
        prev_start = prev_start + size
    return (input_size, net)

# Simulate a run of 'network' given input data 'inp'
def run_net1(network, inp):
    (input_size, ns) = network
    if len(inp) != input_size: raise Exception(f"Input size is {len(inp)}, network requires {input_size}.")
    outp = np.empty(input_size+len(ns))
    outp[:input_size] = inp
    for i, n in enumerate(ns):
        outp[i+input_size] = np.random.binomial(1, apply_node(outp,n))
    return outp

# Output a graphviz visualization of the network
def net_to_dot(network):
    (isz,ns) = network
    print('digraph { ')
    for i in range(isz):
        print(f'  n{i} [ color=blue ]')
    # todo: maybe color all pure output (not in any index) nodes red?
    for i, (ix, ps) in enumerate(ns):
        for x in ix:
            print(f'  n{x} -> n{i+isz}')
    print('}')        

# Generate a random input and run a network
def simulate(network):
    (input_size, _) = network
    return run_net1(network, np.random.choice([0,1], size=input_size))

# Iterate a network over all possible inputs
def iterate(network):
    (input_size, _) = network
    outp = []
    for i in range(2**input_size):
        outp.append(run_net1(network,bindigits(i,input_size)))
    return outp

# Convert an integer 'x' to an array of binary digits
def bindigits(x, sz):
    out = np.empty(shape=sz)
    for i in range(sz):
        out[sz-1-i] = x%2
        x = x//2
    return out

# Testing:
# iterate(mknet(3,6,3))
## net_to_dot(mklayerednet(10,[(10,2),(10,3)]))
