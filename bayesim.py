# using a random bayesian network to generate data parametrized by complexity

from collections import namedtuple
import numpy as np

# A node is a tuple of an array of w indices, and a 2^w array of probabilities
Node = namedtuple('Node', ['inputs', 'probabilities'])

# 'width' is w, i.e. the number of inputs to each node
# 'entropy' is the alpha/beta parameter used to draw probabilities from a beta distribution
# (ie., not to be confused with actual entropy)
# An 'entropy' of zero means a deterministic network, where the same input always give the same outputs in each node.

def mknode(inputwidth, width, first=0, entropy=0):
    ix = np.random.randint(inputwidth, size=width) + first
    if entropy == 0:
        ps = np.random.choice([0,1], size=2**width)
    else:
        ps = np.random.beta(entropy, entropy, size=2**width)
    return Node(ix, ps)

# Look up the node probability based on values in the inputs
def apply_node(indata, node):
    # can't really verify indata has correct size for node (see usage in run_net1)
    inp = indata[node.inputs]
    x = sum(2**i for i, v in enumerate(reversed(inp)) if v) # interpret inp as a base2 index into ps
    return node.probabilities[x]

# A network is an input_size and a list of nodes which can refer back to previous nodes or the input
Network = namedtuple('Network', ['input_size', 'nodes'])

# Create a network given input size, number of (non-input nodes), the input width, and entropy.
def mknet(input_size, output_size, width, entropy=0):
    net = Network(input_size, [])
    for i in range(input_size, input_size+output_size):
        net.nodes.append(mknode(i, width, entropy))
    return net

# ignore entropy for now
def mklayerednet(input_size, layerlist, entropy=0):
    net = Network(input_size, [])
    prev_size = input_size
    prev_start = 0
    for size, width in layerlist:
        for _ in range(size):
            net.nodes.append(mknode(prev_size, width, first=prev_start, entropy=entropy))
        prev_start = prev_start + prev_size
        prev_size = size
    return net

# Simulate a run of 'network' given input data 'inp'
def run_net1(network, inp):
    if len(inp) != network.input_size:
        raise Exception(f"Input size is {len(inp)}, network requires {network.input_size}.")
    outp = np.empty(network.input_size+len(network.nodes))
    outp[:network.input_size] = inp
    for i, n in enumerate(network.nodes):
        outp[i+network.input_size] = np.random.binomial(1, apply_node(outp, n))
    return outp

# Output a graphviz visualization of the network
def net_to_dot(network):
    print('digraph { ')
    for i in range(network.input_size):
        print(f'  n{i} [ color=blue ]')
    # maybe color all pure output (not in any index) nodes red?
    for i, node in enumerate(network.nodes):
        for x in node.inputs:
            print(f'  n{x} -> n{i+network.input_size}')
    print('}')

# Generate a random input and run a network
def simulate(network, iterations=1):
    res = np.empty(shape=(iterations, network.input_size+len(network.nodes)))
    for i in range(iterations):
        res[i] = run_net1(network, np.random.choice([0, 1], size=network.input_size))
    return res

# Iterate a network over all possible inputs
def iterate(network):
    outp = []
    for i in range(2**network.input_size):
        outp.append(run_net1(network, bindigits(i, network.input_size)))
    return outp

# Convert an integer 'x' to an array of binary digits
def bindigits(x, sz):
    out = np.empty(shape=sz)
    for i in range(sz):
        out[sz-1-i] = x%2
        x = x//2
    return out

## Testing:
# net_to_dot(mklayerednet(10,[(10,2),(10,3)]))
# net_to_dot(mknet(10,10,2))
# simulate(mklayerednet(10,[(10,2),(10,3)]))
# iterate(mknet(10,10,2))
