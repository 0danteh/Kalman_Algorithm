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
