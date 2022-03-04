import os
import numpy as np
from tqdm import *
import time
from scipy.special import gamma as spg
from scipy.special import beta as spb
from scipy.special import binom as spbin
from scipy.special import betaln as bln
import scipy.stats as spst
from scipy import optimize



def log_poch(x,n):
    '''
    Input :
        x : <float>
        n : <int>
    Output :
        log_poch<float> log of Pochammer symbol of x of order n, i.e. ln(Gamma(x+n)/Gamma(x)) 
    Further details:
        http://mathworld.wolfram.com/PochhammerSymbol.html
    '''
    return gln(x+n) - gln(x)

def create_folder(path):
    '''
    Input :
        path : <str>
    Output :
        if there is not folder in path, such folder is created
    '''
    
    if not os.path.exists(path):
        os.makedirs(path)
        
def generate_bin_matrix_from_freqs(thetas, N, seed = 0):
    np.random.seed(seed)
    X = np.random.binomial(1, np.repeat(thetas, N)).reshape(len(thetas), N).T
    
    return X

def generate_cts_from_bin_mat(X):
    
    return np.concatenate([[0], np.count_nonzero(X.cumsum(axis=0), axis = 1)])

def count_new_freq(binary_matrix, N, M, r):
    assert N+M<=binary_matrix.shape[0], 'N+M <= # rows'
    # first N rows are training
    
    binary_cumsum = binary_matrix.cumsum(axis = 0)
    yet_to_be_seen =(binary_cumsum[N] == 0)
    seen_with_frequency_at_M = np.array([(binary_cumsum[N+m] == r) for m in range(M)])
    return np.sum(seen_with_frequency_at_M*yet_to_be_seen[np.newaxis,:], axis = 1)