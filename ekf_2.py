import numpy as np
from time import time
from sklearn.metrics import mean_squared_error

class EKF:
    def __init__(self, n_input, n_output, n_hidden, act_fun, sprW=5):

        # Function dimensionalities
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.n_hidden = int(n_hidden)
        # Neuron type
        self.act_fun = act_fun
        if act_fun == 'logistic':
            self.sig = lambda V: 1 / (1 + np.exp(-V))
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif act_fun == 'tanh':
            self.sig = np.tanh
            self.dsig = lambda sigV: 1 - sigV**2
        elif act_fun == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.where(sigV > 0, 1, 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((n_hidden, n_input+1))-1),
                  sprW*(2*np.random.sample((n_output, n_hidden+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None
        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: W[:, -1] + np.dot(W[:, :-1], np.atleast_1d(V).T)
        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda train_input, train_output: np.sqrt(np.mean(np.square(self.feedforward(train_input) - train_output)))
    
    def update(self, X, return_l=False):
        X = np.float64(X)
        if X.ndim == 1 and len(X) > self.n_input:
            X = X[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], X))
        y = self._affine_dot(self.W[1], l)
        if return_l:
            return y, l
        return y
    
    # Assigning labels
    def assign(self,train_input,hbound,lbound=0):
        return np.int64(np.minimum(np.maximum(self.update(train_input, 0),lbound,hbound)))
    
    def ekf_alt(self, train_input, train_output, h, l, step):

        # Compute NN Jacobian using matrix multiplication
        jacobian_D = self.W[1][:, :-1]*self.dsig(l)
        jacobian_H = np.block([[np.kron(jacobian_D, train_input),jacobian_D],[l, 1]]).T
        # Calculate Kalman gain using matrix inversion lemma
        S_inv = np.linalg.inv(self.R) - np.linalg.inv(self.R + jacobian_H @ self.P @ jacobian_H.T) @ jacobian_H @ self.P
        K = self.P @ jacobian_H.T @ S_inv
        # Update weight estimates and covariance using matrix subtraction
        update_dW = step*K @ (train_output - h)
        self.W[0] -= update_dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] -= update_dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P -= K @ jacobian_H @ self.P
        # Adjust covariance if Q is nonzero
        if self.Q_nonzero:
            self.P += self.Q

    def sgd_alt(self, train_input, train_output, h, l, step):
        # compute the error between the hidden layer output and the target output
        error = np.subtract(h, train_output)
        # update the weights of the second layer by subtracting the product of the error, the learning rate, and the hidden layer output with a bias term
        self.W[1] = np.subtract(self.W[1], np.multiply(step, np.hstack((np.matmul(error, l.T), error[:, np.newaxis]))))
        # compute the delta term for the first layer by multiplying the error, the weights of the second layer without the bias term, and the derivative of the sigmoid function applied to the hidden layer output
        delta = np.multiply(np.matmul(error, self.W[1][:, :-1]), self.dsig(l)).flatten()
        # update the weights of the first layer by subtracting the product of the delta, the learning rate, and the input layer output with a bias term
        self.W[0] = np.subtract(self.W[0], np.multiply(step, np.hstack((np.matmul(delta[:, np.newaxis], train_input.T), delta[:, np.newaxis]))))
    
    def train(self, epochs, train_input, train_output, method, Q=None, R=None, P=None, step=1, time_tres=-1):
        # Convert train_input and train_output to float64
        train_input = np.float64(train_input)
        train_output = np.float64(train_output)
        # Initialize variables based on the chosen method
        if method == 'ekf':
            # Extended Kalman Filter (EKF) method
            self.feed = self.ekf_alt
            self.P = P * np.eye(self.nW, dtype=np.float64)  # Initialize covariance matrix
            self.Q = np.zeros((self.nW, self.nW), dtype=np.float64) if Q is None else np.float64(Q)  # Process noise covariance
            self.Q_nonzero = np.any(self.Q)  # Check if Q is non-zero
            self.R = R * np.eye(self.n_output, dtype=np.float64) if np.isscalar(R) else np.float64(R)  # Measurement noise covariance
        elif method == 'sgd':
            # Stochastic Gradient Descent (SGD) method
            self.feed = self.sgd_alt
        else:
            # Raise an error if an invalid method is provided
            raise ValueError("Choose the method to be either 'ekf' or 'sgd'.")
        last_drwdwn = 0  # Initialize last_drawdown variable
        cov = []  # List to store covariance values
        # Loop through epochs
        for epoch in range(epochs):
            # Shuffle training data for each epoch
            train_input_shuffl = [train_input[i] for i in np.random.permutation(len(train_input))]
            train_output_shuffl = [train_output[i] for i in np.random.permutation(len(train_output))]
            y_true_all = []  # List to store true output values
            y_pred_all = []  # List to store predicted output values
            # Iterate through shuffled data
            for i, (train_input, train_output) in enumerate(zip(train_input_shuffl, train_output_shuffl)):
                h, l = self.update(train_input, return_l=True)  # Update and get values
                self.feed(train_input, train_output, h, l, step)  # Update parameters
                # If EKF method, append the trace of covariance matrix to the list
                if method == 'ekf':
                    cov.append(np.trace(self.P))
                # Check if time_tres condition is met or it's the last iteration
                if (time_tres >= 0 and time() - last_drwdwn > time_tres) or (epoch == epochs - 1 and i == len(train_input) - 1):
                    # Print RMSE (Root Mean Squared Error)
                    print(f"RMSE: {np.sqrt(mean_squared_error(np.vstack(y_true_all), np.vstack(y_pred_all)))}")
        return cov  # Return the list of covariance values

