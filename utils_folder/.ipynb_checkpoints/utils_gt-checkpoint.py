import numpy as np
from scipy import stats as spst
'''
    See Supplementary material in 'Using somatic variant richness to 
    mine signals from rare variants in the cancer genome' [2019], Eqn (6);

'''

def draw_power_law(low, high, exp, unif_samples):
    '''
    Generates random samples from a Zipf distribution in the interval [low, high];
    see http://mathworld.wolfram.com/RandomNumber.html for more details;
    Input:
        low float >0
        high float > low
        exp float > 0
        unif_samples array of len K; should be all uniform random numbers in [0,1]
    Output
        K power law distributed random variates
    '''
    return ((high**(exp + 1) - low**(exp+1))*unif_samples + low**(exp+1))**(1/(exp + 1))



def pred_new_gt(N_train, M, sfs, cts, alternative):
    
    '''
    Input:
        N_train < int > number of samples seen so far
        M <int> number of additional samples
        sfs <array of ints> should be of len N
        cts <array of ints> array of len N_train + 1
        alternative < bool > determines which smoothing parametrization to adopt
    Output:
        preds <array> of size N_train + M + 1
    '''
    
    preds = np.zeros(N_train+M)
    preds[:N_train] = cts[:N_train]
    
    if len(sfs) != N_train:
        n__ = N_train - len(sfs)
        sfs = np.concatenate([sfs, np.zeros(n__)])
    
    seen_so_far = cts[N_train]
    signed_sfs = (-1)**np.arange(len(sfs)) * sfs
    
    t_range = np.arange(1,M+1)/N_train
    t_power = np.asarray([t**np.arange(1,N_train+1) for t in t_range])
    
    if M/N_train<=1:
        preds[N_train:] = seen_so_far + np.sum(signed_sfs[np.newaxis, :]*t_power, axis = 1)
    if M/N_train > 1:
        preds[N_train:2*N_train] = seen_so_far + np.sum(signed_sfs[np.newaxis,:]*t_power[:N_train], axis = 1)
        if alternative == True:
            kappa_vec = np.asarray([0.5 * np.log2(N_train * t**2 /(t-1)) for t in t_range[N_train:]], dtype = int)
            theta_vec = 1/(t_range[N_train:]+1)
        else:
            kappa_vec = np.asarray([0.5 * np.log(N_train * t**2 /(t-1))/np.log(3) for t in t_range[N_train:]], dtype = int)
            theta_vec = 2/(t_range[N_train:]+1)
        #probas = np.zeros([M-N_train, len(sfs)])
        for m in range(M-N_train):
            prob = 1-spst.binom.cdf(n=kappa_vec[m], p=theta_vec[m], k=np.arange(len(sfs)))
            preds[2*N_train+m] = seen_so_far + np.sum(signed_sfs*t_power[N_train+m]*prob)
    return preds