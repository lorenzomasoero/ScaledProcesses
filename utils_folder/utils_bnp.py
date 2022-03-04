from utils_all import *
from utils_IBP import *
from utils_GD import *

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

def make_dict(model, seed, N_tot, N_train, true_parameters, width):
    dictiontary = {}
    dictiontary['model'] = model
    dictiontary['seed'] = seed
    dictiontary['N_tot'] = N_tot
    dictiontary['N_train'] = N_train
    dictiontary['true_parameters'] = true_parameters
    dictiontary['width'] = width
    return dictiontary

def synthetic_draw_fit_plot(data_generating_process):
    '''
    Input:
        data_generating_process <dict> with keys: call make_dict
            model <str> either ibp or gd
            true_parameters <array of floats>
            seed <int> for reproducibility
            N_train, N_tot <int> train size, total sample size
            width <float> in [0,1) for credible intervals
    Output:
        opt_p_ibp : fit of ibp params
        opt_p_gd : fit of gd params
    '''
    
    ibp, gd = IBP(), GD()
    seed = data_generating_process['seed']
    N_train, N_tot = data_generating_process['N_train'], data_generating_process['N_tot']
    width = data_generating_process['width']
    
    if data_generating_process['model'] == 'ibp':
        
        # draw training data
        true_model = IBP()
        
        alpha, c, sigma = data_generating_process['true_parameters']
        N_train, N_tot = data_generating_process['N_train'], data_generating_process['N_tot']
        true_model.instantiate_IBP(alpha, c, sigma, N_train, seed = seed)
        counts_train, fa_train = true_model.counts, true_model.fa
        true_model.instantiate_IBP(alpha, c, sigma, N_tot, seed=seed)
        
        
    if data_generating_process['model'] == 'gd':
           
        true_model = GD()
        sigma, tilting = data_generating_process['true_parameters']
        true_model.instantiate_GD(sigma, tilting, N_train, seed = seed)
        counts_train, fa_train = true_model.counts, true_model.fa
        true_model.instantiate_GD(sigma, tilting, N_tot, seed=seed)
        
    counts = true_model.counts
    sfs_train = np.bincount(fa_train)[1:]
    
    # fit
    
    opt_p_ibp = ibp.fit_EFPF_sfs(sfs_train, N_train, num_its=5, num_boots_correction=0, status = 1)
    opt_p_gd = gd.fit_EFPF_sfs(sfs_train, N_train, num_its=5, num_boots_correction=0, status = 1)
    
    # predict
    
    predicted_news_gd = true_model.counts[N_train+1] + gd.mean(N_train, N_tot-N_train, sfs_train.sum(), opt_p_gd) 
    predicted_news_gd = np.concatenate([true_model.counts[:N_train+1], predicted_news_gd])
    
    lo_gd, hi_gd = gd.credible_interval(N_train, N_tot-N_train, sfs_train.sum(), opt_p_gd, width)
    lo_gd, hi_gd = np.concatenate([true_model.counts[:N_train+1], true_model.counts[N_train+1]+lo_gd]), np.concatenate([true_model.counts[:N_train+1], true_model.counts[N_train+1]+hi_gd])
    
    predicted_news_ibp = true_model.counts[N_train+1] + ibp.mean(N_train, N_tot-N_train, opt_p_ibp) 
    predicted_news_ibp = np.concatenate([true_model.counts[:N_train+1], predicted_news_ibp])
    
    lo_ibp, hi_ibp = ibp.credible_interval(N_train, N_tot-N_train, opt_p_ibp, width)
    lo_ibp, hi_ibp = np.concatenate([true_model.counts[:N_train+1], true_model.counts[N_train+1]+lo_ibp]), np.concatenate([true_model.counts[:N_train+1], true_model.counts[N_train+1]+hi_ibp])
    
    
    plt.figure(figsize = (20,7))
    
    plt.plot(true_model.counts, color = 'k', linewidth  =3, label = 'True')

    plt.plot(predicted_news_ibp, color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.', label = '3-IBP')
    plt.fill_between(np.arange(len(lo_ibp)), lo_ibp, hi_ibp, color = 'b', alpha = .2)
    
    plt.plot(predicted_news_gd, color = 'red', alpha = .5, linewidth = '3', linestyle = '--',label = 'GD')
    plt.fill_between(np.arange(len(lo_gd)), lo_gd, hi_gd, color = 'red', alpha = .2)
    
    plt.vlines(x = N_train, ymin = 0, ymax = max(predicted_news_ibp[-1], predicted_news_gd[-1]), color = 'gray', linestyle = '--', label = r'$N$')
    plt.legend()
    plt.title(data_generating_process['model']+str('; ')+str(data_generating_process['true_parameters']), fontsize = 20)
    plt.show()
    
    return opt_p_ibp, opt_p_gd


def plot_from_params(dictionary, opt_p_ibp, opt_p_gd, true_counts, save = False):
    '''
        Input:
            dictionary <dict> created calling make_dict
            opt_p_ibp, opt_p_gd : optimal parameters of ibp and of gd
            true_counts <array of ints> true accumulation curve
            save <str> optional path to save pdf figure
    '''
    
    N_train, N_tot = dictionary['N_train'], dictionary['N_tot']
    sfs = dictionary['sfs']
    width = dictionary['width']
    
    plt.figure(figsize = (20,7))
    
    plt.plot(true_counts, color = 'k', linewidth  =3, label = 'True')
    predicted_news_ibp, predicted_news_gd = [0], [0]
        
    ibp = IBP()

    predicted_news_ibp = true_counts[N_train+1] + ibp.mean(N_train, N_tot-N_train, opt_p_ibp) 
    predicted_news_ibp = np.concatenate([true_counts[:N_train+1], predicted_news_ibp])

    lo_ibp, hi_ibp = ibp.credible_interval(N_train, N_tot-N_train, opt_p_ibp, width)
    lo_ibp, hi_ibp = np.concatenate([true_counts[:N_train+1], true_counts[N_train+1]+lo_ibp]), np.concatenate([true_counts[:N_train+1], true_counts[N_train+1]+hi_ibp])

    plt.plot(predicted_news_ibp, color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.', label = '3-IBP')
    plt.fill_between(np.arange(len(lo_ibp)), lo_ibp, hi_ibp, color = 'b', alpha = .2)
        
    gd = GD()

    predicted_news_gd = true_counts[N_train+1] + gd.mean(N_train, N_tot-N_train, sfs.sum(), opt_p_gd) 
    predicted_news_gd = np.concatenate([true_counts[:N_train+1], predicted_news_gd])

    lo_gd, hi_gd = gd.credible_interval(N_train, N_tot-N_train, sfs.sum(), opt_p_gd, width)
    lo_gd, hi_gd = np.concatenate([true_counts[:N_train+1], true_counts[N_train+1]+lo_gd]), np.concatenate([true_counts[:N_train+1], true_counts[N_train+1]+hi_gd])

    plt.plot(predicted_news_gd, color = 'red', alpha = .5, linewidth = '3', linestyle = '--',label = 'GD')
    plt.fill_between(np.arange(len(lo_gd)), lo_gd, hi_gd, color = 'red', alpha = .2)

    plt.vlines(x = N_train, ymin = 0, ymax = max(predicted_news_ibp[-1], predicted_news_gd[-1]), color = 'gray', linestyle = '--', label = r'$N$')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title(dictionary['title'], fontsize = 20)
    
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', dpi = 1000)
    plt.show()
    
def plot_from_preds(dictionary, preds_ibp, lo_ibp, hi_ibp, preds_gd, lo_gd, hi_gd, true_counts, save = False):
    
    N_train, N_tot = dictionary['N_train'], dictionary['N_tot']
    width = dictionary['width']
    
    plt.figure(figsize = (20,7))
    
    plt.plot(true_counts, color = 'k', linewidth  =3, label = 'True')

    for _ in range(len(preds_ibp)):
        if _ == 0:
            plt.plot(preds_ibp[_], color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.', label = '3-IBP')
        else:
            plt.plot(preds_ibp[_], color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.')
        plt.fill_between(np.arange(len(lo_ibp[_])), lo_ibp[_], hi_ibp[_], color = 'b', alpha = .2)
    
    for _ in range(len(preds_gd)):
        if _ == 0:
            plt.plot(preds_gd[_], color = 'red', alpha = .5, linewidth = '3', linestyle = '--',label = 'GD')
        else:
            plt.plot(preds_gd[_], color = 'red', alpha = .5, linewidth = '3', linestyle = '--')
            
        plt.fill_between(np.arange(len(lo_gd[_])), lo_gd[_], hi_gd[_], color = 'red', alpha = .2)

    plt.vlines(x = N_train, ymin = 0, ymax = max(np.max(preds_ibp), np.max(preds_gd)), color = 'gray', linestyle = '--', label = r'$N$')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title(dictionary['title'], fontsize = 20)
    
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', dpi = 1000)
    plt.show()
    
def plot_from_preds_efpf_reg(dictionary, preds_ibp, lo_ibp, hi_ibp, preds_gd, lo_gd, hi_gd, preds_gd_reg, lo_gd_reg, hi_gd_reg, true_counts, save = False):
    
    N_train, N_tot = dictionary['N_train'], dictionary['N_tot']
    width = dictionary['width']
    
    plt.figure(figsize = (20,7))
    
    plt.plot(true_counts, color = 'k', linewidth  =3, label = 'True')

    for _ in range(len(preds_ibp)):
        if _ == 0:
            plt.plot(preds_ibp[_], color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.', label = '3-IBP')
        else:
            plt.plot(preds_ibp[_], color = 'blue', alpha = .5, linewidth = '3', linestyle = '-.')
        plt.fill_between(np.arange(len(lo_ibp[_])), lo_ibp[_], hi_ibp[_], color = 'b', alpha = .2)
    
    for _ in range(len(preds_gd)):
        if _ == 0:
            plt.plot(preds_gd[_], color = 'red', alpha = .5, linewidth = '3', linestyle = '--',label = 'GD (EFPF)')
        else:
            plt.plot(preds_gd[_], color = 'red', alpha = .5, linewidth = '3', linestyle = '--')
            
        plt.fill_between(np.arange(len(lo_gd[_])), lo_gd[_], hi_gd[_], color = 'red', alpha = .2)
        
    for _ in range(len(preds_gd_reg)):
        if _ == 0:
            plt.plot(preds_gd_reg[_], color = 'orange', alpha = .5, linewidth = '3', linestyle = ':',label = 'GD (regr)')
        else:
            plt.plot(preds_gd_reg[_], color = 'orange', alpha = .5, linewidth = '3', linestyle = ':')
            
        plt.fill_between(np.arange(len(lo_gd_reg[_])), lo_gd_reg[_], hi_gd_reg[_], color = 'orange', alpha = .2)

    plt.vlines(x = N_train, ymin = 0, ymax = max(np.max(preds_ibp), np.max(preds_gd)), color = 'gray', linestyle = '--', label = r'$N$')
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title(dictionary['title'], fontsize = 20)
    
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', dpi = 1000)
    plt.show()
    
    
    
