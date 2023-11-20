import numpy as np

class EKF:
    def __init__(self, nu, ny, nl, neuron, sprW=5):