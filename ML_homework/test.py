import tensorflow as tf

import numpy as np

d = np.floor((np.random.rand(1, 5)))
new = np.empty(d.shape)
new = np.vstack((d, new))
print(new)
print(new.shape)


