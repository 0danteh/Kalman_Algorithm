import numpy as np
from time import time
from sklearn.metrics import mean_squared_error

class EKF:
    def __init__(self, n_input, n_output, n_hidden, activ, sprW=5):
        # Function dimensionalities
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.n_hidden = int(n_hidden)
        # Neuron type
        self.activ = activ
        if activ == 'logistic':
            self.sig = lambda V: 1 / (1 + np.exp(-V))
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif activ == 'tanh':
            self.sig = np.tanh
            self.dsig = lambda sigV: 1 - sigV**2
        elif activ == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.where(np.float64(sigV > 0))
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
    def update(self, X, return_l=False):
        X = np.float64(X)
        if X.ndim == 1 and len(X) > self.n_input:
            X = X[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], X))
        h = self._affine_dot(self.W[1], l)
        if return_l:
            return h, l
        return h
    def ekf_alt(self, x, y, h, l, step):
        # Compute NN Jacobian using matrix multiplication
        jacobian_D = self.W[1][:, :-1]*self.dsig(l)
        jacobian_H = np.block([[np.kron(jacobian_D, x),jacobian_D],[l, 1]]).T
        # Calculate Kalman gain using matrix inversion lemma
        S_inv = np.linalg.inv(self.R) - np.linalg.inv(self.R + jacobian_H @ self.P @ jacobian_H.T) @ jacobian_H @ self.P
        K = self.P @ jacobian_H.T @ S_inv
        # Update weight estimates and covariance using matrix subtraction
        update_dW = step*K @ (y - h)
        self.W[0] = update_dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = update_dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P = K @ jacobian_H @ self.P
        # Adjust covariance if Q is nonzero
        if self.Q_nonzero:
            self.P += self.Q
    def sgd_alt(self, x, y, h, l, step):
        # compute the error between the hidden layer output and the target output
        error = np.subtract(h, y)
        # update the weights of the second layer by subtracting the product of the error, the learning rate, and the hidden layer output with a bias term
        self.W[1] = np.subtract(self.W[1], np.multiply(step, np.hstack((np.matmul(error, l.T), error[:, np.newaxis]))))
        # compute the delta term for the first layer by multiplying the error, the weights of the second layer without the bias term, and the derivative of the sigmoid function applied to the hidden layer output
        delta = np.multiply(np.matmul(error, self.W[1][:, :-1]), self.dsig(l)).flatten()
        # update the weights of the first layer by subtracting the product of the delta, the learning rate, and the input layer output with a bias term
        self.W[0] = np.subtract(self.W[0], np.multiply(step, np.hstack((np.matmul(delta[:, np.newaxis], x.T), delta[:, np.newaxis]))))   
    # Helper function to check and assign matrix params
    def _check_matrix(self, M, n, error_msg):
        if M is None:
            if hasattr(self, 'M') and self.M is not None:
                return self.M
            else:
                raise ValueError(error_msg)
        elif np.isscalar(M):
            return M*np.eye(n, dtype=np.float64)
        else:
            if np.shape(M) != (n, n):
                raise ValueError(error_msg)
            return np.float64(M)

    def train(self, epochs, X, Y, method, Q=None, R=None, P=None, step=1, time_tres=-1):
        # Convert train_input and train_output to float64
        X = np.float64(X)
        Y = np.float64(Y)
        # Check the shape and length of X and Y
        def check_shape(X,Y):
            X_dim = X.ndim
            Y_dim = Y.ndim
            # Get data points
            m = len(X)
            # Make sure X and Y have the same num of data points
            if m != len(Y):
                raise ValueError("X and Y must have the same num!")
            # Check the shape of X
            if X_dim == 1:
                if self.n_input != 1:
                    raise ValueError("X must have one input var")
            else:
                if X.shape[-1] != self.n_input:
                    raise ValueError(f"X must have {self.n_input} input vars")
            # Check the shape of Y
            if Y_dim == 1:
                if self.n_output != 1:
                    raise ValueError("Y must have one input var") 
            # If y has more data points, reshape it as a column vector
            if m>self.n_output:
                Y = Y[:, np.newaxis]  
            else:     
                # If Y is multi-dimensional, check if it has the correct number of output variables
                if Y.shape[-1] != self.n_output:
                    raise ValueError(f"Y must have {self.n_output} vars!")
            return X,Y
        X,Y = check_shape(X,Y)
        # Initialize variables based on the chosen method
        if method == 'ekf':
            # EKF
            self.feed = self.ekf_alt
            # Check for P, Q & R
            self.P = self._check_matrix(P, self.nW, "Initial P not specified")
            self.Q = self._check_matrix(Q, self.nW, "Q must be a float scalar or (nW by nW) array")
            self.R = self._check_matrix(R, self.n_output, "R must be specified for the EKF")
            if np.linalg.matrix_rank(self.R) != len(self.R):
                raise ValueError("R must be positive definite.")
        elif method == 'sgd':
            # Stochastic Gradient Descent (SGD) method
            self.feed = self.sgd_alt
        else: raise ValueError("Choose the method to be either 'ekf' or 'sgd'.")
        # Loop through epochs
        for epoch in range(epochs):
            # Shuffle training data for each epoch
            train_input_shuffl = [X[i] for i in np.random.permutation(len(X))]
            train_output_shuffl = [Y[i] for i in np.random.permutation(len(Y))]
            # Iterate through shuffled data
            for x, y in enumerate(zip(train_input_shuffl, train_output_shuffl)):
                h, l = self.update(x, return_l=True)  # Update and get values
                self.feed(x, y, h, l, step)  # Update parameters
                