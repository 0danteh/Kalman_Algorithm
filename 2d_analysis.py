import numpy as np
import matplotlib.pyplot as plt
from ekf import EKF_NN

# Define the number of points, the number of classes, and the noise level for the data
n_points = 100 # Number of points per class
n_classes = 3 # Number of classes
noise = 0.2 # Noise level

# Generate a linearly spaced array of angles from 0 to 2*pi with the number of points
theta = np.linspace(0, 2 * np.pi, n_points)

# Multiply the angles by a factor that depends on the class label, so that each class forms a spiral with a different number of turns
factors = np.arange(1, n_classes + 1) # Factors for each class
theta = np.outer(theta, factors) # Outer product of theta and factors

# Calculate the x and y coordinates of the points using the sine and cosine of the angles, and add some random noise using a normal distribution
x = theta * np.cos(theta) + np.random.normal(0, noise, theta.shape)
y = theta * np.sin(theta) + np.random.normal(0, noise, theta.shape) 
# Calculate the radius of the spiral using the square root of the sum of the squares of the x and y coordinates
r = np.sqrt(x**2 + y**2)

# Stack the x, y, and r coordinates into a 3D array, and create a 1D array of class labels
X = np.stack([x.flatten(), y.flatten(), r.flatten()], axis=1) # 3D array of points
y = np.repeat(np.arange(n_classes), n_points) # 1D array of labels

# Create a figure and an axes object
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the data using ax.scatter3D()
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral) # Scatter plot of the data
ax.set_xlabel('x') # x-axis label
ax.set_ylabel('y') # y-axis label
ax.set_zlabel('r') # z-axis label
ax.set_title('Spiral data in 3D') # Plot title
plt.show() # Show the plot