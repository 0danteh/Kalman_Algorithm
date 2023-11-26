# Extended Kálmán Filter for Neural Networks

This repo aims at the creation of a manual neural network, found at `def_ekf.py`, that uses the extended Kálmán filter. In order to test such algorithm, I have applied it to a nonlinear system: the double pendulum, getting very interesting results: the EKF has been successfully implemented and was capable of capturing the trend of the physical system.

# Mathematical analysis of the whole project.

The EKF is a recursvie algorithm, meaning that it call itself to solve a problem. It estimates the state of a nonlinear dynamical system from noisy measurements. The Extended Kálmán Filter is composed by two steps: prediction (done by the `predict` function) and update (done within the `_ekf` function). In the prediction step, the state and the state covariance are propagated using a nonlinear model. In the update step, the state and the state covariance are corrected using a measurement and a measurement model. The EKF requires the computation of the Jacobian matrices of the nonlinear models, which are linear approximations of the models around the current state.

The prediction steps goes as follows: $$\hat{x}_{k \mid k-1}=f(\hat{x}\_{k-1 \mid k-1}, u_k)$$

$$P_{k \mid k-1} = F_kP_{k-1 \mid k-1}F_k^T+Q_k$$ where $\hat{x}_{k \mid k-1}$ is the prediction state, $f$ is the nonlinear state transition function, $\hat{x}\_{k-1 \mid k-1}$ is the previous estimate, $u_k$ is the control input, $P\_{k \mid k-1}$ is the predicted state covariance, $F_k$ is the state transition Jacobian matrix, $P\_{k-1 \mid k-1}$ is the previous state covariance estimate, and $Q_k$ is the process noise covariance matrix.

For what concerns the update step, instead, it goes as follows:

$$\tilde{y}_k=z_k-h(\hat{x}\_{k \mid k-1})$$

$$S_k=H_kP_{k \mid k-1}H_k^T+R_k$$

$$K_k=P_{k \mid k-1}H_k^TS_k^{-1}$$

$$\hat{x}_{k \mid k-1}+K_k \tilde{y}_k$$

$$P_{k \mid k} = (l-K_kH_k)P_{k \mid k-1}$$
