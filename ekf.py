import numpy as np
from scipy.special import expit, logit

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return expit(x) * (1 - expit(x))

class EKF_NN:
    def __init__(self, n_input, n_hidden, n_output, Q, R):
        # Initialize network parameters
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_weights = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output
        self.Q = Q # Process noise covariance
        self.R = R # Measurement noise covariance