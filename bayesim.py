# using a random bayesian network to generate data parametrized by complexity

import numpy as np

# A node is a tuple of an array of w indices, and a 2^w array of probabilities
# 'width' is w, i.e. the number of inputs to each node
# 'entropy' is the alpha/beta parameter used to draw probabilities from a beta distribution (ie., not to be confused with actual entropy)
# An 'entropy' of zero means a deterministic network, where the same input always give the same outputs in each node.
def mknode(inputwidth, width, entropy=0):
    ix = np.random.randint(inputwidth, size=width)
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

# Simulate a run of 'network' given input data 'inp'
def run_net1(network, inp):
    (input_size, ns) = network
    if len(inp) != input_size: raise Exception(f"Input size is {len(inp)}, network requires {ins}.")
    out = np.empty(input_size+len(ns))
    out[:input_size] = inp
    for i, n in enumerate(ns):
        out[i+input_size] = np.random.binomial(1, apply_node(out,n))
    return out

# Output a graphviz visualization of the network
def net_to_dot(network):
    return None

# Generate a random input and run a network
def simulate(network):
    (input_size, _) = network
    inp = np.random.choice([0,1], size=input_size)
    return run_net1(network,inp)

# Testing
# n = mknode(4,2,1)
# apply_node_prob(np.array([0,1,0,1], n)
