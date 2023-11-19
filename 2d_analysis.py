from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate a random noisy synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.15, # Noise level
    random_state=42
)

