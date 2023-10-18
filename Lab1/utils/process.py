import math
import numpy as np
import utils.config as config

def sigmoid(x):
    # return 1 / (1 + math.exp(-x))
    return np.tanh(x)

def sig_deriv(x):
    # return x - x ** 2
    return 1 - x ** 2

def init():
    global weights
    global bias
    global output
    global grad

    weights = [[np.random.rand(config.inputs[i]) for j in range(config.nneurons[i])] for i in range(len(config.nneurons))]
    bias    = [np.random.rand(config.nneurons[i])-1 for i in range(len(config.nneurons))]
    output  = [0 for i in range(len(config.nneurons))]
    grad   = [0 for i in range(len(config.nneurons))]

    print(weights)

def forw_prop(result):
    for i in range(len(config.nneurons)):   # layers
        input = np.array(result)
        result = []
        for j in range(config.nneurons[i]): # neurons
            result.append(sigmoid(sum(weights[i][j]*input)+bias[i][j]))
        output[i] = np.array(result)
    return output[-1]

def error(val):
    return sum((output[-1] - val) ** 2)

def back_prop(val):
    grad[-1] = sig_deriv(output[-1])*(val - output[-1])
    for i in range(len(config.nneurons)-2, -1, -1):
        result = []
        for j in range(config.nneurons[i]):
            result.append(sum((weights[i+1][k][j]*grad[i+1][k]) for k in range(config.nneurons[i+1])) * sig_deriv(output[i][j]))
        grad[i] = np.array(result)

def change(val):
    for i in range(len(config.nneurons)):   # layers
        for j in range(config.nneurons[i]): # neurons
            bias[i][j] += config.learning_rate * grad[i][j]
            if i > 0:
                weights[i][j] += config.learning_rate * grad[i][j] * output[i-1]
            else:
                weights[i][j] += config.learning_rate * grad[i][j] * val


def run(index:int, update:bool=False, test:bool=False):
    out = forw_prop(config.test_in[index] if test else config.train_in[index])
    err = error(config.test_out[index] if test else config.train_out[index])
    back_prop(config.test_out[index] if test else config.train_out[index])
    if update:
        change(config.test_in[index] if test else config.train_in[index])
    return (out, err)