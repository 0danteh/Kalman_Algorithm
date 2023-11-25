import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from def_ekf import EKF
import matplotlib as plt

# Define the double pendulum dynamics. 
def double_pendulum(t,y):  
    """
    Parameters:
    - m = mass
    - l = length
    - g = gravity acceleration
    Note: since it's about a double nonlinear pendulum system, number-based labeling conventions
          are used for further distinction of the variables.
    """

    