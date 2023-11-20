import numpy as np

class EKF:
    def __init__(self, n_input, n_output, n_hidden, neuron, sprW=5):

        # Function dimensionalities
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.n_hidden = int(n_hidden)
        # Neuron type
        self.neuron = neuron
        if neuron == 'logistic':
            self.sig = lambda V: 1 / (1 + np.exp(-V))
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = np.tanh
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.maximum(V, 0)
            self.dsig = lambda sigV: np.where(sigV > 0, 1, 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
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
    def assign(self, train_input, hbound, lbound=0):
        return np.int64(np.minimum(np.maximum(self.update(train_input, 0), lbound, hbound)))
    # Calculate the EKF
    def ekf(self, train_input, train_output, h, l, step):
        # Compute NN Jacobian
        jacobian_D = np.multiply(self.W[1][:, :-1], self.dsig(l)).ravel()
        jacobian_H = np.hstack((np.hstack((np.kron(jacobian_D, train_input), jacobian_D[:, np.newaxis])).reshape(self.n_output, self.W[0].size),
                                np.repeat(np.concatenate((l, [1])), self.n_output).reshape(self.n_output, self.n_hidden+1)))
        # Calculate Kalman gain
        S = np.dot(jacobian_H, np.dot(self.P, jacobian_H.T)) + self.R
        K = np.dot(np.dot(self.P, jacobian_H.T), np.linalg.inv(S))
        # Update weight estimates and covariance
        update_dW = step * np.dot(K, train_output - h)
        self.W[0] += update_dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] += update_dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P -= np.dot(K, np.dot(jacobian_H, self.P))
        # Adjust covariance if Q is nonzero
        if self.Q_nonzero:
            self.P += self.Q