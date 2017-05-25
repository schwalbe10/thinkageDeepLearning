import os
import numpy as np
import matplotlib.pylab as plt
import pickle
from mnist import load_mnist


# Plot the loss function
x = np.arange(0, 1, 0.01)
y = -np.log(x)

plt.plot(x, y, color='k', lw=1, linestyle=None)
plt.show()


# Diplay all arrays
# np.set_printoptions(threshold=np.inf)

# Generate a mini-batch
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 3
batch_mask = np.random.choice(train_size, batch_size)
print(x_train[batch_mask])
print(t_train[batch_mask])
