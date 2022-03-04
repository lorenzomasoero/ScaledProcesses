from __future__ import division
import os, sys, random, subprocess, pandas, gzip, math, itertools
import numpy as np
import pandas as pd
import _pickle as cPickle
from operator import itemgetter
from scipy.stats import binom_test
from scipy.stats import chi2_contingency
from scipy.stats import entropy
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import hypergeom
#import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
from cvxopt import matrix, solvers
from utils_all import *

def optimize_unseen_est(path_to_sfs, path_to_histogram, kappa, M):
    '''
        Input:
            path_to_sfs <str>
            path_to_histogram  <str>
            kappa <scalar>0> percentage of histogram to estimate -- 1 corresponds to 1%
            M <int>
        Output :
            preds <array of size M>
    '''
    histx, xLP = unseen_est(path_to_sfs, kappa)
    write_output(histx, xLP, path_to_histogram)
    preds = pred_counts_unseen(path_to_sfs, path_to_histogram, kappa, M)
    return preds
    
def optimize_unseen_est_no_path(sfs, kappa, M, num_its):
    '''
        Input:
            path_to_sfs <array>
            path_to_histogram  <array>
            kappa <scalar>0> percentage of histogram to estimate -- 1 corresponds to 1%
            M <int>
        Output :
            preds <array of size M>
    '''
    
    histogram, xLP = unseen_est_no_path(sfs, kappa)
    preds = pred_counts_unseen_no_path(sfs, histogram, kappa, M)
    return preds

def unseen_est(path_to_sfs, kappa):
    
    f = np.loadtxt(path_to_sfs).astype(int)
    n_samples = len(f)
    up_to = int(n_samples * kappa / 100)
    f = list(f[:up_to])
    
    ########### BASIC CONSTANTS ###################
    gridFactor = 1.01
    maxLPIters = 1000
    xLPmax = len(f)/n_samples
    xLPmin = 1./(n_samples*100)
    N_max = 65000000
    #N_max = 650000000
    
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
    
    return np.array(histx), xLP

def unseen_est_no_path(sfs, kappa):
    
    f = sfs.astype(int)
    n_samples = len(f)
    up_to = int(n_samples * kappa)
    f = list(f[:up_to])
    
    ########### BASIC CONSTANTS ###################
    gridFactor = 1.01
    maxLPIters = 1000
    xLPmax = len(f)/n_samples
    xLPmin = 1./(n_samples*100)
    N_max = 65000000
    #N_max = 650000000
    
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

def write_output(histx, xLP, outname):
    out = open(outname, 'w')
    out.write('\t'.join(['frequency', '# of variants'])+'\n')
    for i in range(len(xLP)):
        out.write('\t'.join([str(xLP[i]), str(histx[i,0])])+'\n')
    out.close()


def pred_counts_unseen(path_to_sfs, path_to_histogram, kappa, M):
    '''
    post processing of Zou algorithm to get the full histogram
    
    Input :
        path_to_sfs
        path_to_histogram
        kappa <float in (0,1)> - determines rare variants
    Output :
        total : <array> (boot_its * num_its * len(checkpoints)) ;  total coincides with true_counts in coordinates 0:'up_to'
    '''

    sfs = np.loadtxt(path_to_sfs)
    N = len(sfs) 
    rare_position = int(N*kappa/100)
    
    # load the output of optimization
    
    unseen_est = pd.read_csv(path_to_histogram, sep='\t') 
    unseen_est_x, unseen_est_h = np.asarray(unseen_est['frequency']), np.asarray(unseen_est['# of variants']) # convert results to arrays
    
    emp_h = sfs[rare_position:]
    emp_x = np.asarray([x/n for x in range(rare_position, len(sfs))])
    
    x_tr, h_tr = np.concatenate((unseen_est_x, emp_x)), np.concatenate((unseen_est_h, emp_h))
    
    return np.asarray([(h_tr*(1-(1-x_tr)**t)).sum() for t in range(N+M)])

def pred_counts_unseen_no_path(sfs, freqs, num_vars, kappa, M):
    '''
    post processing of Zou algorithm to get the full histogram
    
    Input :
        path_to_sfs
        path_to_histogram
        alpha <float in (0,1)> - determines rare variants
    Output :
        total : <array> (boot_its * num_its * len(checkpoints)) ;  total coincides with true_counts in coordinates 0:'up_to'
    '''

    N = len(sfs) 
    rare_position = int(N*kappa)
    
    # load the output of optimization
    
#     unseen_est = pd.read_csv(path_to_histogram, sep='\t') 
#     unseen_est_x, unseen_est_h = np.asarray(histogram['frequency']), np.asarray(histogram['# of variants']) # convert results to arrays
    
    emp_h = sfs[rare_position:]
    emp_x = np.asarray([x/N for x in range(rare_position, len(sfs))])
    
    x_tr, h_tr = np.concatenate((freqs, emp_x)), np.concatenate((num_vars, emp_h))
    
    return np.asarray([(h_tr*(1-(1-x_tr)**t)).sum() for t in range(N+M+1)])

