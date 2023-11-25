import numpy as np
import numpy.linalg as npl
from tqdm import tqdm
from scipy.linalg import block_diag

# Ensure that the input matrix M is valid and has the correct shape. 
# If M is None, create an identity matrix of size n.

def _check_matrix(M,n,error_msg):
 # Check if matrix M is None
    if M is None:
        # Check if M has __dict__ attribute and is not None
        if hasattr('M', '__dict__') and M is not None:
            return M
        else:
            pass
    # Check if M is a scalar
    elif np.isscalar(M):
        # Return scaled identity matrix of size (n, n)
        return M*np.eye(n, dtype=np.float64)
    else:
        # Check if the shape of M is (n, n)        
        if np.shape(M) != (n,n):
            raise ValueError(error_msg)
        return np.float64(M)

# Validate the shapes of input matrices X and Y.
# Ensure they have the correct number of data points, input variables, and output variables.

def validate_shape(X,Y,n_input,n_output):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("U and Y must have the same number of data points.")
    if X.shape[-1] != n_input:
        raise ValueError(f"U must have {n_input} input variables.")
    if Y.shape[-1] != n_output:
        raise ValueError(f"Y must have {n_output} output variables.")
    if Y.ndim == 1 and Y.size>n_output:
        Y=Y.reshape(-1, 1)
    return X, Y

# Compute the outer product of vectors x and y, concatenated with x multiplied by a bias.

def outer_plus_bias(x, y, bias=1):
        return np.hstack((np.outer(x, y), x[:, np.newaxis] * bias))

# Develop the whole EKF Neural Network

class EKF:

    # Compute the sigmoid activation function based on the chosen activation type.
    def sig(self, V):
        if self.activ == 'logistic':
            return (1 + np.exp(-V))**-1
        elif self.activ == 'tanh':
            return np.tanh(V)
        elif self.activ == 'relu':
            return np.clip(V, 0, np.inf)
    
    # Compute the derivative of the sigmoid activation function.       
    def dsig(self, sigV):
        if self.activ == 'logistic':
            return sigV * (1 - sigV)
        elif self.activ == 'tanh':
            return 1 - sigV**2
        elif self.activ == 'relu':
                return np.float64(sigV > 0)

    # Initialize the EKF neural network with given parameters. 
    def __init__(self,n_input,n_output,n_hidden,activ,weight_scale=5):
        
        self.n_input=int(n_input)
        self.n_output=int(n_output)
        self.n_hidden=int(n_hidden)
        self.activ=activ
        if activ not in ['logistic', 'tanh', 'relu']:
            raise ValueError("The 'activ' argument must be 'logistic', 'tanh', or 'relu'.")
        # Random arrays with diff shapes
        def random_weights(shape):
            return weight_scale*(2*np.random.sample(shape)-1)
        # Inilialising weight matrix
        weight_scale=np.float64(weight_scale)
        # Create a list of two arrays with random weights
        self.W=[random_weights((n_hidden,n_input+1)),random_weights((n_output,n_hidden+1))]
        self.num_weights=sum([np.size(w) for w in self.W])
        self.P=None
        self.affine_transform=lambda W,V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

    # Feeding the neural network.
    # Optionally, return the intermediate layer values if specified.
    def update(self,U,return_l=False):
        U=np.float64(U)
        if U.ndim == 1 and len(U) > self.n_input:
            U=U[:, np.newaxis]
        l=self.sig(self.affine_transform(self.W[0], U))
        h=self.affine_transform(self.W[1], l)
        return (h, l) if return_l else h
    
    # Calculate the Jacobian matrix for the EKF algorithm based on input u and intermediate layer values lv
    def jacobian(self, u, l):
        # Compute NN jacobian
        D=(self.W[1][:, :-1]*self.dsig(l)).flatten()
        H=np.hstack((outer_plus_bias(D, u).reshape(self.n_output, self.W[0].size), block_diag(*np.tile(np.concatenate((l, [1])), self.n_output).reshape(self.n_output, self.n_hidden+1))))
        return H
    
    # Update weight estimates and covariance.
    def update_weights_and_cov(self, K, dW):
        # Update weight estimates and covariance
        self.W[0]=self.W[0]+dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1]=self.W[1]+dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P=self.P-np.dot(K, self.H.dot(self.P))
        if self.Q_nonzero: self.P=self.P+self.Q

    # Compute the Kalman gain based on covariance, observation matrix, and measurement noise.
    def kalman_gain(self, P, H, R):
        K = P.dot(H.T).dot(npl.inv(H.dot(self.P).dot(H.T) + self.R))
        return K
    
    # Extended Kalman Filter for the neural network.
    def _ekf(self, u, y, h, l, step):
        # Update the network
        self.H=self.jacobian(u,l)
        # Kalman gain
        K=self.kalman_gain(self.P, self.H, self.R)
        # Update weight estimates and covariance
        dW=step*K.dot(y-h)
        self.update_weights_and_cov(K, dW)

    # Train the neural network for a given number of epochs.
    def train(self, n_epochs, U, Y, P=None, Q=None, R=None, step=1):

        """
        Parameters:
        - n_epochs: Number of training epochs.
        - U: Input training data.
        - Y: Output training data.
        - P: Initial covariance matrix for the EKF algorithm.
        - Q: Covariance matrix for the process noise in the EKF algorithm.
        - R: Covariance matrix for the measurement noise in the EKF algorithm.
        - step: Learning step size.

        Note: This method uses the specified training method and updates the network weights accordingly.
        """

        U=np.float64(U)
        Y=np.float64(Y)
        U,Y=validate_shape(U,Y,self.n_input,self.n_output)
        # Initialise EKF
        self.feed = self._ekf
        if P is None:
            if self.P is None:
                raise ValueError("Initial P not specified.")
        else:
            self.P=_check_matrix(P, self.num_weights, "P must be a float scalar or (num_weights by num_weights) array.")
        self.Q=_check_matrix(Q, self.num_weights, "Q must be a float scalar or (num_weights by num_weights) array.")
        if np.any(self.Q): self.Q_nonzero=True
        else: self.Q_nonzero=False
        self.R=_check_matrix(R, self.n_output, "R must be a float scalar or (n_output by n_output) array.")
        if npl.matrix_rank(self.R) != len(self.R):
            raise ValueError("R must be positive definite.")

        # Start the training
        for epoch in range(n_epochs):
            shuffl=np.random.permutation(len(U))
            train_input_shuffled=U[shuffl]
            train_output_shuffled=Y[shuffl]
            pbar = tqdm(train_input_shuffled, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            # Update the neural networks and train
            for i, (u,y) in enumerate(zip(pbar, train_output_shuffled)):
                h, l=self.update(u, return_l=True)
                self.feed(u, y, h, l, step)