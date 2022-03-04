from scipy.stats import binom
from cvxopt import matrix, solvers
import numpy as np
from scipy.special import binom as spbinom



def unseen_est(n_samples, sfs, kappa):
    
    f = sfs.astype(int)
    #n_samples = len(f)
    up_to = int(n_samples * kappa)
    f = list(f[:up_to])
    
    ########### BASIC CONSTANTS ###################
    gridFactor = 1.01
    maxLPIters = 1000
    xLPmax = len(f)/n_samples
    xLPmin = 1./(n_samples*100)
    N_max = 65000000
    
    ########### SETTING UP THE LP ###################
    fLP = f + [0]*int(np.ceil(np.sqrt(len(f))))
    szLPf = len(fLP)
    xLP = xLPmin*np.power(gridFactor, np.arange(0, np.ceil(np.log(xLPmax/xLPmin)/np.log(gridFactor))+1))
    szLPx = np.max(xLP.shape)
    
    ## set up the objective function
    objf = np.zeros((1, szLPx + 2*szLPf))
    objf[0, np.arange(szLPx, szLPx + 2*szLPf, 2)] = 1./np.sqrt(np.array(fLP) + 1)
    objf[0, np.arange(szLPx+1, szLPx + 2*szLPf, 2)] = 1./np.sqrt(np.array(fLP) + 1)
    
    ## set up the inequality constraints corresponding to the moment matching
    ## first 2*szLPf are for moment matching, next szLPx+2*szLPf for >=0, last for <= N_max
    A = np.zeros((2*szLPf+szLPx+2*szLPf+1, szLPx+2*szLPf))  
    b = np.zeros((2*szLPf+szLPx+2*szLPf+1, 1))
    
    rv_list = [binom(n_samples, x) for x in xLP]
    # moment matching constraints
    for i in range(szLPf):
        A[2*i, np.arange(szLPx)] = [rv.pmf(i+1) for rv in rv_list]
        A[2*i+1, np.arange(szLPx)] = -A[2*i, np.arange(szLPx)]
        A[2*i, szLPx+2*i] = -1
        A[2*i+1, szLPx+2*i+1] = -1
        b[2*i, 0] = fLP[i]
        b[2*i+1, 0] = -fLP[i]
    
    # >= 0 constraints
    for i in range(szLPx+2*szLPf):
        A[i+2*szLPf,i] = -1
        b[i+2*szLPf,0] = 0
        
    # <= N_max constraint
    A[-1,range(szLPx)] = 1
    b[-1,0] = N_max
    
        
    ## set up the equality constraints
    Aeq = np.zeros((1, szLPx+2*szLPf))
    Aeq[0, range(szLPx)] = xLP
    beq = np.sum(np.array(f)*(1+np.arange(len(f))))/n_samples
    
    ########### RUNNING THE LP ###################
    
    solvers.options['show_progress'] = False
    
    ## rescaling for better conditioning
    for j in range(np.max(xLP.shape)):
        A[:,j] = A[:,j]/xLP[j]
        Aeq[0,j] = Aeq[0,j]/xLP[j]
    
    #return objf, A, b, szLPf, szLPx, xLP
    sol = solvers.lp(matrix(objf.T), matrix(A), matrix(b), matrix(Aeq), matrix(beq))    
    #res = linprog(list(objf[0]), A_ub = A, b_ub = list(b.T[0]), A_eq = Aeq, b_eq = [beq] , options = {'maxiter': maxLPIters})
    
    ## remove the scaling
    histx = np.array(sol['x'])[0:szLPx]
    histx = [histx[i]/xLP[i] for i in range(szLPx)]
    hh = np.array(histx)
    return hh.reshape(len(hh)), xLP


def pred_counts_unseen(sfs, kappa, N, M):
    '''
    post processing of Zou algorithm to get the full histogram
    
    Input :
        path_to_sfs
        path_to_histogram
        kappa <float in (0,1)> - determines rare variants
    Output :
        total : <array> (boot_its * num_its * len(checkpoints)) ;  total coincides with true_counts in coordinates 0:'up_to'
    '''

    rare_position = int(N*kappa)    
    unseen_est_h, unseen_est_x = unseen_est(N, sfs, kappa)
    
    emp_h = sfs[rare_position:]
    emp_x = np.asarray([x/N for x in range(rare_position, len(sfs))])
    
    x_tr, h_tr = np.concatenate((unseen_est_x, emp_x)), np.concatenate((unseen_est_h, emp_h))
    
    return np.asarray([(h_tr*(1-(1-x_tr)**t)).sum() for t in range(N+M+1)])

def new_variants_with_frequency_unseen(bins, densities, N, M, r):
    '''
        Predict number of variants with frequency r or up to r (less_or_equal = True)
    
        Input :
            bins <array> values in [0,1] of variants' frequencies
            densities <array> corresponding "height" of the histrogram h(x) for each x in bins
        Output :
            total : <array> (boot_its * num_its * len(checkpoints)) ;  total coincides with true_counts in coordinates 0:'up_to'
    '''
        
    total_news = np.asarray([(densities*(spbinom(t,r))*bins**(r)*(1-bins)**(t-r)).sum() for t in range(N+1,N+M+1)])
#     print(total_news.shape)
    correction = (densities*(spbinom(N,r))*bins**(r)*(1-bins)**(N-r)).sum()
    prediction = np.zeros([N+M+1])
    prediction[N+1:] = total_news - correction
    return prediction
