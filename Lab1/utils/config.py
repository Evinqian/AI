import numpy as np

nneurons = [1, 1, 1]
inputs = [1] + nneurons[:-1]
learning_rate = 0.005
max_epoch = 5000


train_data  = np.linspace(-np.pi, np.pi, 1001)
target_data = np.sin(np.linspace(-np.pi, np.pi, 1001))