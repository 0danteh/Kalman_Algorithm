import numpy as np
import numpy.linalg as npl
from tqdm import tqdm
from scipy.linalg import block_diag

def _check_matrix(M,n,error_msg):
    if M is None:
        if hasattr('M', '__dict__') and M is not None:
            return M
        else:
            pass
    elif np.isscalar(M):
        return M*np.eye(n, dtype=np.float64)
    else:
        if np.shape(M) != (n,n):
            raise ValueError(error_msg)
        return np.float64(M)

def validate_shape(X,Y,n_input,n_output):
    # check for the same number of data points
    if X.shape[0] != Y.shape[0]:
        raise ValueError("U and Y must have the same number of data points")
    # Check for the input variables to avoid shape-related problems
    if X.shape[-1] != n_input:
        raise ValueError(f"X must have {n_input} input vars")
    # Check for Y now
    if Y.shape[-1] != n_output:
        raise ValueError(f"Y must have {n_output} output vars")
    # Reshape Y as a column vector if one-dimensional
    if Y.ndim == 1 and Y.size>n_output:
        Y=Y.reshape(-1,1)
    return X,Y

def outer_plus_bias(x,y,bias=1):
    return np.hstack((np.outer(x,y),x[:,np.newaxis]*bias))

class EKF:
    def sigm(self,V):
        if self.activ == 'logistic':
            return (1 + np.exp(-V))**-1
        elif self.activ == 'tanh':
            return np.tanh(V)
        elif self.activ == 'relu':
            return np.clip(V,0,np.inf)
        
    def deriv_sigm(self, sigmV):
        if self.activ == 'logistic':
            return sigmV*(1-sigmV)
        elif self.activ == 'tanh':
            return 1-sigmV**2
        elif self.activ == 'relu':
            return np.float64(sigmV>0)

    def __init__(self, n_input, n_output, n_hidden, activ, weight_scale=5):
        
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
        weight_scale = np.float64(weight_scale)
        # Create a list of two arrays with random weights
        self.W = [random_weights((n_hidden, n_input+1)), random_weights((n_output,n_hidden+1))]
        self.num_weights = sum([np.size(w) for w in self.W])
        self.P = None
        self._affine_transf = lambda W,V: np.dot(np.atleast_1d(V), W[:,:-1].T)+W[:,-1]

    # Feeding the neural network
    def update(self,U,return_l=False):
        U=np.float64(U)
        # Ensuring X has 2 dims
        if U.ndim==1 and len(U)>self.n_input:
            U=U[:,np.newaxis]
        l = self.sigm(self._affine_transf(self.W[0],U))
        h = self._affine_transf(self.W[1],l)
        return (h,l) if return_l else h
    
    # Calculate the jacobian for the EKF algorithm
    def jacobian(self,u,l):
        # Compute the jacobian
        D=(self.W[1][:,:-1]*self.deriv_sigm(l)).flatten()
        H=np.hstack((outer_plus_bias(D,u).reshape(self.n_output, self.W[0].size), block_diag(*np.tile(np.concatenate((l,[1])), self.n_output).reshape(self.n_output,self.n_hidden+1))))
        return H
    
    def update_weights_and_cov(self,K,dW):
        # Update weights and covariances
        self.W[0]=self.W[0]+dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1]=self.W[1]+dW[self.W[0].size].reshape(self.W[1].shape)
        self.P=self.P-np.dot(K,self.H.dot(self.P))
        if self.Q_nonzero: self.P=self.P+self.Q

    # Compute the kalman gain
    def kalman_gain(self,P,H,R):
        K=P.dot(H.T).dot(npl.inv(H.dot(self.P).dot(H.T)+self.R))
        return K
    
    def _ekf(self,u,y,h,l,step):
        # Update the network
        self.H=self.jacobian(u,l)
        # Get the Kalman gain
        K=self.kalman_gain(self.P,self.H,self.R)
        # Update the weights and covariances
        dW=step*K.dot(y-h)
        self.update_weights_and_cov(K,dW)

    def train(self,n_epochs,U,Y,P=None,Q=None,R=None,step=1):
        U=np.float64(U)
        Y=np.float64(Y)
        # Check if the shapes are as eUpected
        U,Y=validate_shape(U,Y,self.n_input,self.n_output)
        # Initialise the EKF algorithm
        self.feed=self._ekf
        # Check for the correct shapes
        if P is None:
            if self.P is None:
                raise ValueError("P needs to be specified.")
        else:
            self.P=_check_matrix(P,self.num_weights,"P must be a float scalar or (num_weights by num_weights) array.")
        self.Q=_check_matrix(Q,self.num_weights,"Q must be a float scalar or (num_weights by num_weights) array.")
        if np.any(self.Q): self.Q_nonzero=True
        else: self.Q_nonzero=False
        self.R=_check_matrix(R,self.n_output,"R must be a float scalar or (n_output by n_output) array.")
        if npl.matrix_rank(self.R)!=len(self.R):
            raise ValueError("R must be definite and positive.")

        # Start the training
        for epoch in range(n_epochs):
            shuffl=np.random.permutation(len(U))
            train_input_shuffled=U[shuffl]
            train_output_shuffled=Y[shuffl]
            pbar=tqdm(train_input_shuffled, desc=f"Epoch {epoch+1}/{n_epochs}", 
                      unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            # Update the neural networks and train
            for i, (u,y) in enumerate(zip(pbar, train_output_shuffled)):
                h,l=self.update(u,return_l=True)
                self.feed(u,y,h,l,step)

# Get some training data from the simulation of a nonlinear system, the Lorenz Attractor!
dt = 0.01  # physical resolution
tf = 100  # experiment duration
T = np.arange(0, tf, dt, dtype=np.float64)  # time record
X = np.zeros((len(T), 3), dtype=np.float64)  # state record
Xdot = np.zeros_like(X)  # state derivative record
x = np.array([1, 1, 1], dtype=np.float64)  # initial condition
for i, t in enumerate(T):
    X[i] = np.copy(x)  # record
    Xdot[i] = np.array((10*(x[1]-x[0]),
                        x[0]*(28-x[2])-x[1],
                        x[0]*x[1]-2.6*x[2]))  # dynamic
    x = x + Xdot[i]*dt  # step simulation
per = 0.01  # training data sampling period
skip = int(per/dt)

# Create and train KNN
knn = EKF(n_input=3, n_output=3, n_hidden=20, activ='relu')
knn.train(n_epochs=1, U=X[::skip], Y=Xdot[::skip], P=0.5, R=0.1)


# Use KNN to simulate system from same initial condition
Xh = np.zeros_like(X)
xh = X[0]
for i, t in enumerate(T):
    Xh[i] = np.copy(xh)
    xh = xh + knn.update(xh)*dt

# Plot the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2], label='True trajectory', linewidth=0.5)
ax.plot(Xh[:, 0], Xh[:, 1], Xh[:, 2], label='Simulated trajectory with KNN', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()