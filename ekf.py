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
        # Calculate the inverse of the process noise covariance matrix Q
        try:
            self.prior_cov = np.linalg.inv(Q)
        except np.linalg.LinAlgError:
            raise ValueError("Q matrix is not invertible")
        self.R = R # Measurement noise covariance matrix (n_output x n_output)
        if not isinstance(R, np.ndarray) or R.shape != (n_output, n_output):
            raise ValueError("Invalid dimensions for R matrix")
        # Initialize network weights randomly using a normal distribution
        self.w = np.random.normal(0, 0.01, self.n_weights)
        # Initialize error covariance matrix
        self.P = np.eye(self.n_weights) * 1e-3 # Identity matrix with a small diagonal value

        def predict(self, x):
            # Predict the output of the network given an input vector
            x = np.append(x, 1) # Add bias term to input vector (n_input + 1 x 1)
            z = np.dot(x, self.w[:self.n_input + 1].reshape(self.n_input + 1, self.n_hidden)) # Input to hidden layer (n_hidden x 1)
            h = 1 / (1 + np.exp(-z)) # Hidden layer output using sigmoid activation function (n_hidden x 1)
            h = np.append(h, 1) # Add bias term to hidden layer output (n_hidden + 1 x 1)
            y = np.dot(h, self.w[self.n_input + 1:].reshape(self.n_hidden + 1, self.n_output)) # Input to output layer (n_output x 1)
            y = 1 / (1 + np.exp(-y)) # Output layer output using sigmoid activation function (n_output x 1)
            return y