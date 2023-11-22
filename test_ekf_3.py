import numpy as np
import numpy.linalg as npl
from time import time
from sklearn.metrics import mean_squared_error
from scipy.linalg import block_diag

def _check_matrix(M,n,error_msg):
    