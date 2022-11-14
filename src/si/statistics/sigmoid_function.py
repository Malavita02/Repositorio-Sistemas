import numpy as np

def sigmoid_funtion(x):
    sigmoid = 1/(1 + np.exp(-x))
    return sigmoid