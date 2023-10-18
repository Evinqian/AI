import math
import numpy as np
import matplotlib.pyplot as plt
import utils.config as config
import utils.process as process
import random

print("Net has %d layers with size" % len(config.nneurons), config.nneurons)

np.random.seed(42)

process.init()
train_err = []
test_err = []
for i in range(config.max_epoch):
    if i in config.l_rates.keys():
        config.learning_rate = config.l_rates[i]
        print("Learning rate: %f" % config.learning_rate)
    error = 0
    out = [0 for i in range(len(config.train_in))]
    t = list(range(len(config.train_in)))
    random.shuffle(t)
    for j in t:
        out[j], err = process.run(j, True, False)
        error += err
    train_err.append(error/len(config.train_in))
    error = 0
    out = [0 for i in range(len(config.test_in))]
    t = list(range(len(config.test_in)))
    random.shuffle(t)
    for j in t:
        out[j], err = process.run(j, False, True)
        error += err
    test_err.append(error/len(config.test_in))
    if i % 100 == 0:
        print("epoch %d : %f %f" % (i, train_err[-1], test_err[-1]))
        plt.scatter(config.test_in, 1.25*np.array(out), s=1)
        plt.scatter(config.test_in, 1.25*config.test_out, s=1)
        plt.show()

plt.scatter(list(range(len(train_err))), train_err, s=1)
plt.scatter(list(range(len(test_err))), test_err, s=1)
plt.show()
print(process.weights)
print(process.bias)