import numpy as np
import numpy.linalg as npl
from time import time
from sklearn.metrics import mean_squared_error
from scipy.linalg import block_diag

def _check_matrix(M,n,error_msg):
    if M is None:
        if hasattr('M', '__dict__') and M is not None:
            return M
        else:
            raise ValueError(error_msg)
    elif np.isscalar(M):
        return M*np.eye(n, dtype=np.float64)
    else: 
        if np.shape(M) != (n,n):
            raise ValueError(error_msg)
        return np.float64(M)

class KF_EKF:

    def sigm(self,V):
        if self.activ == 'logistic':
            return 1/(1+np.exp(-V))
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
            return np.where(np.float64(sigmV>0))

    def __init__(self, n_input, n_output, n_hidden, activ, weight_scale=5):
        
        self.n_input=int(n_input)
        self.n_output=int(n_output)
        self.n_hidden=int(n_hidden)
        self.activ=self.activ
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
    def update(self,X,return_l=False):
        X=np.float64(X)
        # Ensuring X has 2 dims
        if X.ndim==1 and len(X)>self.n_input:
            X=X[:,np.newaxis]
