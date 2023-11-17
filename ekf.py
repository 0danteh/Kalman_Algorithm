import numpy as np
from scipy.special import expit, logit

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

