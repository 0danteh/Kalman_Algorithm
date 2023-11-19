from sklearn.datasets import make_classification
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

# Convert the data to a numpy array (reference)
data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

# Plot the data
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Random Noisy Synthetic Dataset")
plt.show()

# Create an instance of the EKF_NN class
ekf_nn = EKF_NN(
    n_input=2, # Number of input nodes
    n_hidden=10, # Number of hidden nodes
    n_output=1, # Number of output nodes
    # Create the Q and R matrices by multiplying the scalar values by identity matrices of the appropriate sizes
    Q = 0.01*np.eye(71),
    R = 0.1*np.eye(1),
) 