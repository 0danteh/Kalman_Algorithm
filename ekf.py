import numpy as np
from scipy.special import expit, logit

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return expit(x) * (1 - expit(x))

class EKF_NN:
    def __init__(self, n_input, n_hidden, n_output, Q, R):
        # Initialize network parameters
        self.n_input = n_input # Number of input nodes
        self.n_hidden = n_hidden # Number of hidden nodes
        self.n_output = n_output # Number of output nodes
        self.n_weights = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output # Total number of weights
        self.Q = Q # Process noise covariance matrix (n_weights x n_weights)
        if not isinstance(Q, np.ndarray) or Q.shape != (self.n_weights, self.n_weights):
        raise ValueError("Invalid dimensions for Q matrix")
        self.R = R # Measurement noise covariance matrix (n_output x n_output)

        # Initialize network weights randomly
        self.init_weights()

        # Initialize error covariance matrix
        self.P = np.eye(self.n_weights) * 1e-3 # Identity matrix with a small diagonal value