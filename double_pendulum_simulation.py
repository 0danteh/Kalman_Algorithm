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
     (-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2) -
      2*np.sin(theta1-theta2)*m2*(omega2**2*l2+omega1**2*l1*np.cos(theta1-theta2)))/
     (l1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2))),
     omega2,
     (2*np.sin(theta1-theta2)*(omega1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+
      omega2**2*l2*m2 * np.cos(theta1 - theta2))) /
     (l2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
    ]
    return dydt

# --- Simulate the double pendulum ---

# Simulate the time span
t_span=(0,10)
# Initial conditions 
y0=[np.pi/4, 0, np.pi/2, 0]
sol=solve_ivp(double_pendulum, t_span, y0, t_eval=np.arange(0,10,0.01))

# Extract the simulated data
t_sim=sol.t
y_sim=sol.y.t

# Adding noise
np.random.seed(42)
y_sim_noise = y_sim + 0.1 * np.random.randn(*y_sim.shape)

# Data Preprocessing
ekf=EKF(n_input=4,n_output=4,n_hidden=20,activ='relu')

# Standardise the data
scaler=StandardScaler()
y_sim_noise_scaled=scaler.fit_transform(y_sim_noise)

# Training
per=0.01 # Data sampling period
nth=int(per/0.01) # How much to skip
ekf.train(n_epochs=1, U=y_sim_noise_scaled[::nth], Y=y_sim_noise_scaled[::nth], P=0.5, R=0.1)

# Data prediction list
y_pred_list=[]

# Perform the prediction
for i in range(len(t_sim)):
    # Predictions from the scaled input
    y_pred_scaled=ekf.update(y_sim_noise_scaled[i])
    # Inverse transform of the predictions to the original scale
    y_pred=scaler.inverse_transform(y_pred_scaled.reshape(1,-1))
    # Append the values to the list
    y_pred_list.append(y_pred.flatten())

# Convert the list to a numpy array
y_pred_array=np.array(y_pred_list)
