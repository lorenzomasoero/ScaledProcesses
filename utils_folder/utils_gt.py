import numpy as np
from scipy import stats as spst
'''
    See Supplementary material in 'Using somatic variant richness to 
    mine signals from rare variants in the cancer genome' [2019], Eqn (6);

'''

def missed_gt(N, M, sfs, alternative = 0):
    
    
    assert len(sfs)<N+1, 'Too many entries in the sfs; 0-th entry should be # things observed once; last entry # things observed N times'
    
    signed_sfs = (-1)**np.arange(len(sfs)) * sfs
    t = M/N
    t_power = t**np.arange(1,len(sfs)+1)
    if M <= N:
        preds = np.sum(signed_sfs*t_power)
        vars_ = np.sum(sfs*t_power**2)
    else:
        if alternative == True:
            kappa = int(0.5 * np.log2(N * t**2 /(t-1)))
            theta = 1/(t+1)
        else:
            kappa = int(0.5 * np.log(N * t**2 /(t-1))/np.log(3))
            theta = 2/(t+1)
        prob = 1-spst.binom.cdf(n=kappa, p=theta, k=np.arange(len(sfs)))
        preds = np.sum(signed_sfs*t_power*prob)
        vars_ = np.sum(np.abs(signed_sfs)*t_power**2*prob**2)
    return preds, vars_

def predict_gt(N, M, sfs, cts, alternative = 0): 
    '''
        Input :
            sfs array; sfs[0] is  # things observed once
            M <int> extrapolation size
            order <int> jackknife order

    '''

    preds, vars_ = np.zeros(N+M+1), np.zeros(N+M+1)
    preds[:N+1] = cts[:N+1]
    preds_vars = [missed_gt(N, m, sfs, alternative) for m in range(1,M+1)]
    preds[N+1:] = cts[N] + [p[0] for p in preds_vars]
    vars_[N+1:] = [p[1] for p in preds_vars]
    
    return preds, vars_