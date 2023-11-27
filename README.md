# Extended Kálmán Filter for Neural Networks

This repo aims at the creation of a manual neural network, found at `def_ekf.py`, that uses the extended Kálmán filter. In order to test such algorithm, I have applied it to a nonlinear system: the double pendulum, getting very interesting results: the EKF has been successfully implemented and was capable of capturing the trend of the physical system.

# Mathematical analysis the Extended Kálmán Filter.

The EKF is a recursvie algorithm, meaning that it call itself to solve a problem. It estimates the state of a nonlinear dynamical system from noisy measurements. The Extended Kálmán Filter is composed by two steps: prediction (done by the `predict` function) and update (done within the `_ekf` function). In the prediction step, the state and the state covariance are propagated using a nonlinear model. In the update step, the state and the state covariance are corrected using a measurement and a measurement model. The EKF requires the computation of the Jacobian matrices of the nonlinear models, which are linear approximations of the models around the current state. The Kálmán filter equations are derived from the Bayesian inference framework, which updates the posterios probability of the state given the measurements using the prior probability and the likelihood function.

The prediction equations are two: 

The **predicted state estimate**, which is the result of applying the nonlinear state transition function to the previous state estimate and the control input:

$$\hat{x}_{k \mid k-1}=f(\hat{x}\_{k-1 \mid k-1}, u_k)$$

The **predicted state covariance estimate**, which is the result of propagating the previous state covariance estimate through the state transition Jacobian matrix and adding the process noise covariance matrix:

$$P_{k \mid k-1} = F_kP_{k-1 \mid k-1}F_k^T+Q_k$$ where $\hat{x}_{k \mid k-1}$ is the prediction state, $f$ is the nonlinear state transition function, $\hat{x}\_{k-1 \mid k-1}$ is the previous estimate, $u_k$ is the control input, $P\_{k \mid k-1}$ is the predicted state covariance, $F_k$ is the state transition Jacobian matrix, $P\_{k-1 \mid k-1}$ is the previous state covariance estimate, and $Q_k$ is the process noise covariance matrix. 

These equations are derived from the **Taylor series expansion** of the nonlinear state transition function, which approximates it by a linear function plus higher-order terms. The state transition Jacobian matrix is the first-order partial derivative of the state transition function with respect to the state.

For what concerns the update step, instead, the equations go as follows:

The **measurement residual** or **innovation**, which is the difference between the actual measurement and the predicted measurement based on the predicted state:

$$\tilde{y}_k=z_k-h(\hat{x}\_{k \mid k-1})$$

The **innovation covariance**, which is the sum of the predicted measurement covariance and the measurement noise covariance:

$$S_k=H_kP_{k \mid k-1}H_k^T+R_k$$

The **Kalman gain**, which is the product of the predicted state covariance, the measurement model matrix, and the inverse of the innovation covariance:

$$K_k=P_{k \mid k-1}H_k^TS_k^{-1}$$

The **updated state estimate**, which is the sum of the predicted state estimate and the product of the Kalman gain and the measurement residual:

$$\hat{x}_{k \mid k-1}+K_k \tilde{y}_k$$

The **updated state covariance estimate**, which is the product of the identity matrix minus the product of the Kalman gain and the measurement model matrix, and the predicted state covariance:

$$P_{k \mid k} = (I-K_kH_k)P_{k \mid k-1}$$ where $\tilde{y}_k$​ is the measurement innovation, $z_k$​ is the measurement, $h$ is the nonlinear measurement function, $S_k$​ is the innovation covariance, $H_k$​ is the measurement Jacobian matrix, $R_k$​ is the measurement noise covariance matrix, $K_k$​ is the Kalman gain, $\hat{x}\_{k \mid k}$​ is the updated state estimate, and $\hat{P}\_{k \mid k}$​ is the updated state covariance estimate.

As already alluded, a fundamental part of the EKF is done through the Jacobian matrices, which are linear approximations of the models around the current state. The Jacobian matrix of a function is a matrix that contains the partial derivatives of the function with respect to its variables. The neural network consists of two layers: a hidden layer and an output layer. The hidden layer computes a linear transformation of the input followed by a nonlinear activation function, while the output layer computes another linear transformation of the hidden layer output. The activation function can be either logistic, tanh, or relu, depending on the parameter; going as follows:

$$\text{Logistic: } \sigma(x) = \frac{1}{1+e^{-x}} \\
\text{ Tanh: } tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}} \\
\text{ Relu: } relu(x) = max(0,x)$$

The output of the neural network is given by:

$$h(x,u) = W_1l(x,u)+b_1$$ Where $W_1$ and $b_1$ are the weights and biases of the output layer, and $l(x,u)$ is the output of the hidden layer, which is given by:

$$l(x,u)= \sigma(W_0u+b_0)$$ where $W_0$ and $b_0$ are the weights and biases of the hidden layer, and $\sigma$ is the activation function. The Jacobian matrix of $h$ with respect to $x$ is then given by:

$$J_h(x,u) = 
\begin{bmatrix}
    \frac{\partial h_1}{\partial x_1}(x,u) & \frac{\partial h_1}{\partial x_2}(x,u) & \dots & \frac{\partial h_1}{\partial x_n}(x,u) \\
    \frac{\partial h_2}{\partial x_1}(x,u) & \frac{\partial h_2}{\partial x_2}(x,u) & \dots & \frac{\partial h_2}{\partial x_n}(x,u) \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    \frac{\partial h_m}{\partial x_1}(x,u) & \frac{\partial h_m}{\partial x_2}(x,u) & \dots & \frac{\partial h_m}{\partial x_n}(x,u)
\end{bmatrix}$$ (https://quicklatex.com/cache3/9e/ql_085b31014ab30a2a69602de5e2205d9e_l3.png) where $n$ is the size of the state vector $x$, and m is the size of the output vector $h$. To compute the elements of this matrix, we need to use the chain rule of differentiation. For example, the element in the first row and first column is given by: 

$$\frac{\partial h_1}{\partial x_1}(x,u) = \frac{\partial h_1}{\partial l_1}(x,u) \frac{\partial l_1}{\partial x_1}(x,u) + \frac{\partial h_1}{\partial l_2}(x,u) \frac{\partial l_2}{\partial x_1}(x,u)+ \dots + \frac{\partial h_1}{\partial l_k}(x,u) \frac{\partial l_k}{\partial x_1}(x,u)$$ where $k$ is the size of the hidden layer output $l$. The partial derivatives of $h_1​$ with respect to $l_i$​ are simply the elements of the first row of $W_1$​, that is, $W_{1,1,i}$​. The partial derivatives of $l_i$​ with respect to $x_j​$ are given by:

$$\frac{\partial l_i}{\partial x_j}(x,u)= \sigma'(W_{0,i,j}u+b_{0,i})W_{0,i,j}$$ where $W_{0,i,j}$ and $b_{0,i}$ are the elements of the $i$-th row of $W_0$ and $b_0$, respectively. Therefore, the element in the first row and first column of $J_h​(x,u)$ is given by:

$$\frac{\partial h_1}{\partial x_1}(x,u)=\sum_{i=1}^kW_{1,1,i} \sigma'(W_{0,i,1}u+b_{0,i})W_{0,i,1}$$ The other elements of the Jacobian matrix can be computed in a similar way, by using the appropriate rows and columns of $W_1$​ and $W_0​$. The code for computing the Jacobian matrix is given by the function `jacobian` in the code.

# Mathematical formulation for the Double Pendulum.

The double pendulum consists of two pendulums connected in series. The angles of the pendulums are denoted by $\theta_1$ and $\theta_2$, and their angular velocity are denoted by $\omega_1$ and $\omega_2$, respectively. THe masses of the pendulum bobs are $m_1$ and $m_2$, and the lengths of the pendulum rodsd are $l_1$ and $l_2$. The gravitational acceleration is denoted by $g$.

The equations of motion for a double pendulum are derived using Lagrangian mechanics. The Langragian ($L$) is calculated as: $$L=T-U$$ where $T$ is the kinetic energy, while $U$ is the potential energy. For that, the kinetic energy of each pendulum bob is given by:

$$T_i= \frac{1}{2}m_i(\dot\theta_i)^2$$ where $i$ is the index of the pendulum bob. The potential energy of each pendulum bob, instead, is given by:

$$U_i=m_igl_i(1- \cos{\theta_i})$$ The Lagrangian for the double pendulum, then becomes: $$L=T_1+T_2-U_1-U_2$$ Therefore, the equations of motion are obtained by applying the Euler_Lagrange equation, such that: 

$$\frac{d}{dt}(\frac{\partial L}{\partial \dot\theta_i})- \frac{\partial L}{\partial \theta_i}=0$$ From this, it is possible to evaluate a system of four second-order ordinary differential equations, one for each variable $\theta_1$, $\omega_1$  , $\theta_2$ , $\omega_2$



