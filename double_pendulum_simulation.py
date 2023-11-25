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

    m1 = 1.0 
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    g = 9.8

    theta1,omega1,theta2,omega2 = y
    