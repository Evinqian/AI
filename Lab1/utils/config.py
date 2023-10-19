import numpy as np

nneurons = [1, 2, 1]
inputs = [1] + nneurons[:-1]
learning_rate = 0.05
l_rates = {0:0.05, 20:0.1, 500:0.05, 1000:0.02, 1500:0.005}
max_epoch = 2000


train_in  = np.linspace(-np.pi, np.pi, 1001)
train_out = 0.5*np.sin(train_in)

test_in = np.array([(train_in[i] + train_in[i+1])/2 for i in range(0, len(train_in)-1, 5)])
test_out = 0.5*np.sin(test_in)