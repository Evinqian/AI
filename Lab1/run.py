import math
import numpy as np
import utils.config as config
import utils.process as process

print("Net has %d layers with size" % len(config.nneurons), config.nneurons)

np.random.seed(42)

process.init()

process.forw_prop(config.train_data[42])
print(process.error(42))