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
        
        def jacobian(self, x):
            # Compute the Jacobian matrix of the network output with respect to the weights
            x = np.append(x, 1) # Add bias term to input vector (n_input + 1 x 1)
            z = np.dot(x, self.w[:self.n_input + 1].reshape(self.n_input + 1, self.n_hidden)) # Input to hidden layer (n_hidden x 1)
            h = 1 / (1 + np.exp(-z)) # Hidden layer output using sigmoid activation function (n_hidden x 1)
            h = np.append(h, 1) # Add bias term to hidden layer output (n_hidden + 1 x 1)
            y = np.dot(h, self.w[self.n_input + 1:].reshape(self.n_hidden + 1, self.n_output)) # Input to output layer (n_output x 1)
            y = 1 / (1 + np.exp(-y)) # Output layer output using sigmoid activation function (n_output x 1  
            # Compute the partial derivatives of the output layer output with respect to the weights
            dy_dw = np.zeros((self.n_output, self.n_weights)) # Partial derivatives matrix (n_output x n_weights)
            for i in range(self.n_output):
                for j in range(self.n_hidden + 1):
                    dy_dw[i, self.n_input + 1 + i * (self.n_hidden + 1) + j] = y[i] * (1 - y[i]) * h[j] # dy_i / dw_    
            # Compute the partial derivatives of the hidden layer output with respect to the weights
            dh_dw = np.zeros((self.n_hidden, self.n_weights)) # Partial derivatives matrix (n_hidden x n_weights)
            for i in range(self.n_hidden):
                for j in range(self.n_input + 1):
                    dh_dw[i, i * (self.n_input + 1) + j] = h[i] * (1 - h[i]) * x[j] # dh_i / dw_    
            # Compute the Jacobian matrix using the chain rule
            H = np.zeros((self.n_output, self.n_weights)) # Jacobian matrix (n_output x n_weights)
            for i in range(self.n_output):
                for j in range(self.n_weights):
                    for k in range(self.n_hidden):
                        H[i, j] += dy_dw[i, self.n_input + 1 + i * (self.n_hidden + 1) + k] * dh_dw[k, j] # H_ij = dy_i / dw_j + sum_k (dy_i / dw_k) * (dh_k / dw_j 
            return H
        
        def update(self, x, y):
            # Update the network weights and the error covariance matrix based on the input and output vectors
            y_pred = self.predict(x) # Predict the output of the network given the input vector
            H = self.jacobian(x) # Compute the Jacobian matrix of the network output with respect to the weights
            # Compute the innovation, innovation covariance, and Kalman gain matrices
            nu = y - y_pred # Innovation vector (n_output x 1)
            Q = np.dot(np.dot(H, self.P), H.T) + self.R # Innovation covariance matrix (n_output x n_output)
            try:
                S_inv = np.linalg.inv(Q) # Inverse of the innovation covariance matrix
            except np.linalg.LinAlgError:
                raise ValueError("S matrix is not invertible")
            K = np.dot(np.dot(self.P, H.T), S_inv) # Kalman gain matrix (n_weights x n_output)
            # Update the network weights and the error covariance matrix
            self.w = self.w + np.dot(K, nu) # Weight update vector (n_weights x 1)
            I = np.eye(self.n_weights) # Identity matrix (n_weights x n_weights)
            self.P = np.dot(np.dot(I - np.dot(K, H), self.P), (I - np.dot(K, H)).T) + np.dot(np.dot(K, self.R), K.T) # Error covariance update matrix (n_weights x n_weights)