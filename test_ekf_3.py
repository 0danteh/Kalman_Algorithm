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
    