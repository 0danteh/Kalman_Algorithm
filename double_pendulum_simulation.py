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
    
    dydt = [
     omega1,
     (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) -
      2 * np.sin(theta1 - theta2) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))) /
     (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))),
     omega2,
     (2 * np.sin(theta1 - theta2) * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) +
      omega2**2 * l2 * m2 * np.cos(theta1 - theta2))) /
     (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    ]
    return dydt

# --- Simulate the double pendulum ---

# Simulate the time span
t_span=(0,10)
# Initial conditions 
y0=[np.pi/4, 0, np.pi/2, 0]
