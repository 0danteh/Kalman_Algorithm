import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from def_ekf import EKF
import matplotlib as plt

# Define the double pendulum dynamics. 