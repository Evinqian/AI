import math
import numpy as np
import utils.config as config

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sig_deriv(x):
    res = sigmoid(x)
    return res - res ** 2

def init():
    global weights
    global bias
    global output

    weights = [[np.random.rand(config.inputs[i]) for j in range(config.nneurons[i])] for i in range(len(config.nneurons))]
    bias    = [np.random.rand(config.nneurons[i])-1 for i in range(len(config.nneurons))]
    output  = [0 for i in range(len(config.nneurons))]

    print(weights)

def forw_prop(result):
    for i in range(len(config.nneurons)):   # layers
        input = np.array(result)
        result = []
        for j in range(config.nneurons[i]): # neurons
            result.append(sigmoid(sum(weights[i][j]*input)+bias[i][j]))
        output[i] = np.array(result)
        print("layer %d," % i, result)

def error(pos):
    return sum((output[-1] - config.target_data[pos]) ** 2)