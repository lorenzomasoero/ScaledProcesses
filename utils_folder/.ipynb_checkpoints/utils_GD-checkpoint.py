import numpy as np
import scipy.stats as spst
from scipy.special import beta as spb
from scipy import optimize
from scipy.special import gammaln as gln
from scipy.special import betaln as bln
from scipy.special import binom as spbin
from tqdm import tqdm_notebook
import time

class GD():
    
    def __init__(self):
        
        return
        
    def instantiate_GD(self, beta, sigma, tilting, N, seed=0, store_matrix = False):
        
        self.beta = beta
        self.sigma = sigma
        self.tilting = tilting
        
        self.N = N
        if store_matrix ==  True:
            self.counts, self.unseen, self.fa, self.sfs, self.bin_mat = self.draw_GD(beta, sigma, tilting, seed, True)
        else:     
            self.counts, self.unseen, self.fa, self.sfs = self.draw_GD(beta, sigma, tilting, seed)
            
    def draw_GD(self, beta, sigma, tilting, seed = 0, store_matrix = False):
        
        np.random.seed(seed)
        K = 0
        # let's draw the number of new features
        unseen = np.zeros(1+self.N, dtype = int)
        
        for n in range(1,self.N+1):
            
            pn1 = self.GD_pnm(n-1,1,beta,sigma) # n is how many seen so far; so always n-1
            news_ = np.random.negative_binomial(K+tilting+1, 1-pn1)
            K+= news_
            unseen[n] = news_
        counts = unseen.cumsum()

        # now we populate the matrix

            
        fa = np.zeros([counts[-1]])
        low = counts[1]
        fa[:low] = 1
        
        if store_matrix ==  True:
            bin_mat = np.zeros([self.N, counts[-1]], dtype = int)
            bin_mat[0,:low] = 1
        
        for n in range(1,self.N):

            # let's populate old features
            hi = counts[n+1]
            biases_0 = fa[:low]-sigma
            biases_1 = n+1-sigma - fa[:low]
            J_old = np.random.beta(biases_0, biases_1)
            xnk_old = np.random.binomial(n=1, p=J_old)
            fa[:low] += xnk_old
            fa[low:hi]=1
            
            if store_matrix == True:
                bin_mat[n, :low] = xnk_old
                bin_mat[n,low:hi] = 1
            
            low = hi
        sfs = np.bincount(fa.astype(int))[1:]
        
        if store_matrix ==  True: 
            return counts, unseen, fa.astype(int), sfs, bin_mat
        
        return counts, unseen, fa.astype(int), sfs
        
    def GD_pnm(self, N, M, beta, sigma):
        '''
        Input:
            N : number of samples seen so far
            M : number of additional samples
            K : number of features seen so far
            sigma : tail parameter
        Output:
            p_{N,M,\sigma} : probability of success in Negative Binomial draw
        '''
        
        phi = sigma * spb(1-sigma, N+np.arange(1, M+1)).cumsum()
        xi = sigma * spb(1-sigma, np.arange(1, N+1)).sum()
        pnm = phi/(beta+phi+xi)        
        return pnm
    
    def GD_pnm_freq(self, N, M, r, beta, sigma):
        '''
        Input:
            N : number of samples seen so far
            M : number of additional samples
            K : number of features seen so far
            sigma : tail parameter
        Output:
            p_{N,M,\sigma} : probability of success in Negative Binomial draw
        '''
        
        rho = sigma * spbin(np.arange(1,M+1),r) *spb(r-sigma, N+np.arange(1, M+1)-r+1)
        xi = sigma * spb(1-sigma, np.arange(1, N+1)).sum()
        pnm = rho/(beta+rho+xi)        
        return pnm
    
    def credible_interval(self, N, M, K, parameters, width):
        
        beta, sigma, tilting = parameters
        pr = self.GD_pnm(N, M, beta, sigma)
        lo, hi = spst.nbinom.interval(width, n = K+1+tilting, p = 1 - pr)
        return lo, hi
    
    def mean(self, N, M, K, parameters):
        
        beta, sigma, tilting = parameters
        pr = self.GD_pnm(N, M, beta, sigma)
        return spst.nbinom.mean(n = K+1+tilting, p = 1 - pr)
    
    
    def median(self, N, M, K, parameters):
        
        beta, sigma, tilting = parameters
        pr = self.GD_pnm(N, M, beta, sigma)
        return spst.nbinom.median(n = K+1+tilting, p = 1 - pr)
    
    def variance(self, N, M, r, K, parameters):
        beta, sigma, tilting = parameters
        pr = self.GD_pnm(N, M, r, beta, sigma)
        return spst.nbinom.var(n = K+1+tilting, p = 1 - pr)
    
    def accumulation_curve(self, bin_mat):
        bin_mat_cumsum = bin_mat.cumsum(axis = 0)
        return np.concatenate([[0],np.count_nonzero(bin_mat_cumsum, axis = 1)]).astype(int)
    
    def shuffle_accumulation_curves(self, bin_mat, num_shuffles, seed):
        N = len(bin_mat)
        accumulation_curves = np.zeros([num_shuffles+1, 1+N])
        np.random.seed(seed)
        accumulation_curves[0] = self.accumulation_curve(bin_mat)
        for s in range(num_shuffles):
            bin_mat_sh = bin_mat[np.random.permutation(N)]
            accumulation_curves[s+1] = self.accumulation_curve(bin_mat_sh)
            
        return accumulation_curves
    
    ############ SAME QUANTITIES BUT FOR FREQUENCIES
    
    def credible_interval_freq(self, N, M, r, K, parameters, width):
        
        beta, sigma, tilting = parameters
        pr = self.GD_pnm_freq(N, M, r, beta, sigma)
        lo, hi = spst.nbinom.interval(width, n = K+1+tilting, p = 1 - pr)
        return lo, hi
    
    def mean_freq(self, N, M, r, K, parameters):

        beta, sigma, tilting = parameters
        pr = self.GD_pnm_freq(N, M, r, beta, sigma)
        return spst.nbinom.mean(n = K+1+tilting, p = 1 - pr)
    
    def median_freq(self, N, M, r, K, parameters):
        
        beta, sigma, tilting = parameters
        pr = self.GD_pnm_freq(N, M, r, beta, sigma)
        return spst.nbinom.median(n = K+1+tilting, p = 1 - pr)
    
    def variance_freq(self, N, M, r, K, parameters):
        beta, sigma, tilting = parameters
        pr = self.GD_pnm_freq(N, M, r, beta, sigma)
        return spst.nbinom.var(n = K+1+tilting, p = 1 - pr)
    
    def accumulation_curve_freq(self, bin_mat, r):
        bin_mat_cumsum = bin_mat.cumsum(axis = 0)
        
        return np.concatenate([[0], np.array([np.sum(bin_mat_cumsum[n] == r, dtype = int) for n in range(bin_mat_cumsum.shape[0])])])
    
    def shuffle_accumulation_curves_freq(self, bin_mat, r, num_shuffles, seed):
        N = len(bin_mat)
        accumulation_curves = np.zeros([num_shuffles+1, 1+N])
        np.random.seed(seed)
        accumulation_curves[0] = self.accumulation_curve_freq(bin_mat, r)
        for s in range(num_shuffles):
            bin_mat_sh = bin_mat[np.random.permutation(N)]
            accumulation_curves[s+1] = self.accumulation_curve_freq(bin_mat_sh, r)
            
        return accumulation_curves
    
    '''

        GD : FITTING WITH EFPF

    '''



    def fit_EFPF(self, sfs, N, num_its=10, bnds = ((0,10000), (.01,.999), (0,10000),), status = False):
        
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
    


    def make_EFPF(self, sfs, N):
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
        def EFPF(params):
            '''
                Takes as input parameters and returns discrepancy of true counts and predicted counts;
            '''
            beta, sigma, tilting = params
            cost = K*np.log(sigma) - (K+tilting+1) * np.log(beta+sigma*spb(1-sigma, np.arange(1,N+1)).sum())  + gln(K+tilting+1) - gln(tilting+1) + (tilting+1)*np.log(beta) + np.inner(sfs,bln(np.arange(1,len(sfs)+1)-sigma, N - np.arange(1,len(sfs)+1) + 1))
            return - cost

        return EFPF
    

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
        FITTING VARIANCE
    '''
    
    def make_cost_function_variance(self, N, M, K, original_parameters, tolerance, weight_mean, lambda_mean, lambda_variance, target_variance, norm):
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
        
        target_mean = K+self.mean(N, M, K, original_parameters)

        def cost_function(params):

            '''
                Takes as input parameters and returns discrepancy of true counts and predicted counts;
            '''
            predicted_mean = K + self.mean(N, M, K, params) 
            delta_mean = (predicted_mean - target_mean)/(len(target_mean)*target_mean)

            new_variance = self.variance(N, M, K, params)[-1]
            delta_variance = (new_variance - target_variance)/target_variance
            cost = weight_mean * np.linalg.norm(delta_mean, ord = norm) + (1-weight_mean)*np.abs(delta_variance) + lambda_mean*(tolerance<delta_mean[-1]) + lambda_variance*(delta_variance<0)

            return cost

        return cost_function


    def regression_variance(self, N, M, K, original_parameters, tolerance, weight_mean, lambda_mean, lambda_variance, target_variance, num_its, norm, status):
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

        cost = self.make_cost_function_variance(N, M, K, original_parameters, tolerance, weight_mean, lambda_mean, lambda_variance, target_variance, norm)
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
        return optimal_params_[opt_ind] #, optimal_values_
    
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
    
    '''
        FIND LARGEST p
    '''
    
    def find_largest_p(self, N, M, K, original_parameters, num_its, status=False):
        
        target_mean = self.mean(N,M,K,original_parameters)[-1]
        # now set c = 0
        p_max= (target_mean/(1+K))/(1+target_mean/(1+K))
        return self.regression_gd_pnm(N, M, K, original_parameters, p_max, num_its, status)