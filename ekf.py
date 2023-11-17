import numpy as np
from keract import sigmoid, sigmoid_prime

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
