import numpy as np
import scipy.stats as spst
from scipy.special import beta as spb
from scipy import optimize
from scipy.special import gammaln as gln
from scipy.special import betaln as bln
from scipy.special import binom as spbin
from tqdm import tqdm_notebook
import time

class BB():
    
    '''
        Class to fit and instantiate beta-Bernoulli model [see Ionita-Laza 2009 PNAS]
    '''
    
    def __init__(self):
        
        return
        
    def instantiate_BB(self, alpha, beta,  N, K_ub =int(1e5), seed=0, store_matrix = False):
        
        self.alpha = alpha
        self.beta = beta
        self.K_ub = K_ub
        self.N = N
        if store_matrix ==  True:
            self.counts, self.fa, self.sfs, self.bin_mat = self.draw_BB(alpha, beta, seed, True)
        else:     
            self.counts, self.fa, self.sfs = self.draw_BB(alpha, beta, seed)
            
    def draw_BB(self, alpha, beta, N, seed = 0, store_matrix = False):
        '''
        Input:
            alpha <float >0> first parameter of beta prior
            beta <float >0> second parameter of beta prior
            seed <int> RNG
            store_matrix <bool>
        Output:
            counts <array> shape N+1: # distinct variants
            fa <array> shape K: frequency for each variant
            sfs <array> site frequency spectrum
        '''
        
        np.random.seed(seed)
        # let's draw the population frequencies
        freqs = np.random.beta(alpha, beta, size = self.K_ub)
        thetas = freqs[freqs>0]
        X = np.random.binomial(1, np.repeat(thetas, self.N).reshape(self.N, len(thetas)))
        bin_mat = np.random.binomial(1, np.repeat(thetas, self.N).reshape(self.N, len(thetas)))
        counts = self.accumulation_curve(bin_mat)
        fa = bin_mat.sum(axis = 0)
        sfs = np.bincount(fa)[1:]
        
        if store_matrix ==  True:
            return counts, unseen, fa.astype(int), sfs, bin_mat
        return counts, fa.astype(int), sfs
    
    def accumulation_curve(self, bin_mat):
        bin_mat_cumsum = bin_mat.cumsum(axis = 0)
        return np.concatenate([[0],np.count_nonzero(bin_mat_cumsum, axis = 1)]).astype(int)

    

    
    '''

        Beta : fitting with likelihood

    '''



    def fit_EFPF(self, sfs, N, num_its=10, status = False):
        bnds = ((0,10000), (.01,.999), (0,10000),)
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its,3])

        cost = self.make_EFPF(sfs, N)
        if status == True:
            for it in tqdm_notebook(range(num_its)):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind]
    


    def make_beta_likelihood(self, sfs, N):
        '''
        Input : 
            sfs < array of ints, len S<=N > site frequency spectrum 
            N < int > total number of observation; 
        Output :
            cost_function <function>; this is Eqn (***); cost function from using norm on true_cts with n_lo = from_ and n_hi = up_to
                Input : params sigma, c
                Output : scalar loss
        '''
        sfs = np.asarray(sfs)
        K = sfs.sum()
        def beta_likelihood(params):
            '''
                Takes as input parameters and returns discrepancy of true counts and predicted counts;
            '''
            return - cost

        return beta_likelihood
    

    '''
        GD: FITTING MEAN WITH REGRESSION
    '''
    
    def make_cost_function(self, train_counts, from_, up_to, norm):
        '''
        Input : 
            train_counts < array of ints, len N > true distinct counts 
            from_ < int 0<= from < N > index of lowest sample from which we count J_{n_{low}} with our predictions
            up_to < int ; from_ < up_to <= N> index of highest sample to which we match the count J_{n_{hi}} with our predictions
            norm = int -- norm to be used
        Output :
            cost_function <function>; this is Eqn (***); cost function from using norm on true_cts with n_lo = from_ and n_hi = up_to
                Input : params beta, c, sigma
                Output : scalar loss
        '''
        # train_counts[0] should be 0
    
        def cost_function(params):

            '''
                Takes as input parameters and returns discrepancy of true counts and predicted counts;
            '''
            predicted = train_counts[from_] + self.mean(from_, up_to - from_, train_counts[from_], params) 
            delta = predicted - train_counts[from_:up_to]
            cost = np.linalg.norm(delta, ord = norm)
            return cost

        return cost_function


    def regression(self, train_counts, num_its, norm, status):
        '''
        Input :
            train_counts < array of ints ; len N > 
            num_its < int > number of times to optimization is performed
            norm < int > loss chosen
            status <bool> print status
        Output :
            optimal_params <array> (boot_its * num_its * 3)
            optimal_values <array> (boot_its * num_its)
        '''

        bnds =  ((0,10000), (.01,.999), (0,10000),) # fixed bounds of support of 3-param beta process;
        optimal_values_, optimal_params_ = np.zeros(num_its), np.zeros([num_its,3])
        N = len(train_counts)
        from_, up_to = int(N * 1 / 5), N # n_lo and n_hi ; different choices could be done as long as 0<=n_lo<n_hi<=N
        cost = self.make_cost_function(train_counts, from_, up_to, norm)
        if status == True:
            for it in tqdm_notebook(range(num_its)):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind]

    
    '''
        REGRESSION p
    '''
    
    def make_cost_function_gd_pnm(self, N, M, K, target_p):
        
        def cost_function(params):
            beta, sigma = params
            cost = np.abs(self.GD_pnm(N, M, beta, sigma)[-1] - target_p)

            return cost
        return cost_function

    def regression_gd_pnm(self, N, M, K, original_params, target_p, num_its, status):
        '''
        Input :
            train_counts < array of ints ; len N > 
            num_its < int > number of times to optimization is performed
            norm < int > loss chosen
            status <bool> print status
        Output :
            optimal_params <array> (boot_its * num_its * 3)
            optimal_values <array> (boot_its * num_its)
        '''

        bnds =  ((0,10000), (.01,.999),) # fixed bounds of support of 3-param beta process;
        optimal_values_, optimal_params_ = np.zeros(num_its), np.zeros([num_its,3])
        
        target_mu = self.mean(N,M,K,original_params)[-1]
        cost = self.make_cost_function_gd_pnm(N, M, K, target_p)
        if status == True:
            for it in tqdm_notebook(range(num_its)):
                devol = optimize.differential_evolution(cost, bnds)
                alp, sig = devol.x
                p_new = self.GD_pnm(N,M,alp,sig)[-1]
                c = target_mu*(1-p_new)/p_new -K-1
                optimal_params_[it] = alp, sig, c 
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                alp, sig = devol.x
                p_new = self.GD_pnm(N,M,alp,sig)[-1]
                c = target_mu*(1-p_new)/p_new -K-1
                optimal_params_[it] = alp, sig, c 

                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind] 
    