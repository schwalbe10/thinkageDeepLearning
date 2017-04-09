import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
