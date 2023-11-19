import numpy as np

class EKF_NN:
    def __init__(self, n_input, n_hidden, n_output, Q, R):
        # Initialize network parameters
        self.n_input = n_input # Number of input nodes
        self.n_hidden = n_hidden # Number of hidden nodes
        self.n_output = n_output # Number of output nodes
        self.n_weights = (n_input + 1) * n_hidden + (n_hidden + 1) * n_output # Total number of weights
        self.Q = Q # Process noise covariance matrix
        self.R = R # Measurement noise covariance matrix
        # Initialize network weights randomly using a normal distribution
        self.w = np.random.normal(0, 0.01, self.n_weights)
        # Initialize error covariance matrix
        self.P = np.eye(self.n_weights) * 1e-3 # Identity matrix with a small diagonal value

    def h(self, w, x):
        # Add bias term to input vector
        x = np.append(x, 1)
        # Compute the hidden layer output
        z = np.dot(x, w[:self.n_input + 1].reshape(self.n_input + 1, self.n_hidden))
        h = 1 / (1 + np.exp(-z))
        # Add bias term to hidden layer output
        h = np.append(h, 1)
        # Compute the output layer output
        y = np.dot(h, w[self.n_input + 1:].reshape(self.n_hidden + 1, -1))
        y = 1 / (1 + np.exp(-y))
        # Return the output layer output
        return y  
        
    def jacobian(self, x):
        # Compute the Jacobian matrix of the network output with respect to the weights
        x = np.append(x, 1) # Add bias term to input vector
        z = np.dot(x, self.w[:self.n_input + 1].reshape(self.n_input + 1, self.n_hidden)) # Input to hidden layer
        h = 1 / (1 + np.exp(-z)) # Hidden layer output using sigmoid activation function
        y = np.dot(h, self.w[self.n_input + 1:].reshape(self.n_hidden + 1, self.n_output)) # Input to output layer
        y = 1 / (1 + np.exp(-y)) # Output layer output using sigmoid activation function
        # Compute the partial derivatives of the output layer output with respect to the weights
        dy_dw = np.zeros((self.n_output, self.n_weights)) # Partial derivatives matrix
        for i in range(self.n_output):
            for j in range(self.n_hidden + 1):
                dy_dw[i, self.n_input + 1 + i * (self.n_hidden + 1) + j] = y[i] * (1 - y[i]) * h[j] 
        # Compute the partial derivatives of the hidden layer output with respect to the weights
        dh_dw = np.zeros((self.n_hidden, self.n_weights)) # Partial derivatives matrix
        for i in range(self.n_hidden):
            for j in range(self.n_input + 1):
                dh_dw[i, i * (self.n_input + 1) + j] = h[i] * (1 - h[i]) * x[j]
        # Compute the Jacobian matrix using the chain rule
        H = np.zeros((self.n_output, self.n_weights)) # Jacobian matrix
        for i in range(self.n_output):
            for j in range(self.n_weights):
                for k in range(self.n_hidden):
                    H[i, j] += dy_dw[i, self.n_input + 1 + i * (self.n_hidden + 1) + k] * dh_dw[k, j] # H_ij = dy_i / dw_j + sum_k (dy_i / dw_k) * (dh_k / dw_j 
        return H
        
    def predict(self, x):
        # Compute the network output y using the observation function h
        y = self.h(self.w, x)
        # Compute the Jacobian matrix F using the state transition function f
        F = self.f(self.w)
        # Update the network weights w using the state transition function f
        self.w = F @ self.w
        # Update the error covariance matrix P using the Jacobian matrix F and the process noise covariance matrix Q
        self.P = F @ self.P @ F.T + self.Q
        # Return the network output y and the Jacobian matrix F
        return y, F
    
    def update(self, y_meas):
        # Compute the network output y_pred using the observation function h
        y_pred = self.h(self.w, self.x)
        # Compute the Jacobian matrix H using the partial derivative of h with respect to the network weights w
        H = self.jacobian(self.w, self.x)
        # Compute the innovation and innovation covariance
        residual = y_meas - y_pred
        S = H @ self.P @ H.T + self.R
        # Compute the Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # Update the network weights w and the error covariance matrix P using the Kalman gain and the innovation
        self.w = self.w + K @ residual
        self.P = (np.eye(self.n_weights) - K @ H) @ self.P  

    def rmse(self, X, D):
        # Compute the root mean squared error between the predicted output and the actual output
        n_samples = X.shape[0] # Number of samples
        error = 0 # Initialize error
        for i in range(n_samples):
            x = X[i, :] # Input vector for sample i
            d = D[i, :] # Output vector for sample i
            y = self.predict(x) # Predicted output vector for sample i
            error += np.sum((y - d) ** 2) # Sum of squared errors for sample i
        error = error / n_samples # Mean squared error
        error = np.sqrt(error) # Root mean squared error
        return error

    def train(self, X, Y, epochs, lr):
        # Loop over the epochs
        for epoch in range(epochs):
            # Shuffle the training data
            perm = np.random.permutation(X.shape[0])
            X = X[perm]
            Y = Y[perm]
            # Loop over the samples
            for x, y in zip(X, Y):
                # Predict the network output and the Jacobian matrix F
                y_pred, F = self.predict(x)
                # Calculate the error
                error = y - y_pred
                # Update the network weights and the error covariance matrix
                self.update(error, F, lr)
        # Return the final network weights and the error covariance matrix
        return self.w, self.P