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
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(self.feedforward(U) - Y)))
    