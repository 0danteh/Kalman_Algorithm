import numpy as np

class EKF:
    def __init__(self, n_input, n_output, n_hidden, neuron, sprW=5):

        # Function dimensionalities
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.n_hidden = int(n_hidden)
    