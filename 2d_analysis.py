from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from ekf import EKF_NN

# Generate a random noisy synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.15, # Noise level
    random_state=42
)

# Create an instance of the EKF_NN class
ekf_nn = EKF_NN(
    n_input=2, # Number of input nodes
    n_hidden=10, # Number of hidden nodes
    n_output=1, # Number of output nodes
    # Create the Q and R matrices by multiplying the scalar values by identity matrices of the appropriate sizes
    Q = 0.01 * np.eye((2 + 1) * 10 + (10 + 1) * 1), # Q 
    R = 0.1*np.eye(1),
) 

# Train the EKF_NN on the data
ekf_nn.train(X, y, epochs=10)