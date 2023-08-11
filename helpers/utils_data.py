"""
utilities for generating and plotting the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def gen_randmtx(p, q, scale_range=(1,3), diag=False):
    """ generate a p by q random matrix"""
    mtx = np.random.choice([-1,1],size=(p,q))*np.random.uniform(scale_range[0],scale_range[1],size=(p,q))
    if diag:
        mtx = np.diag(np.diag(mtx))
    return mtx

def get_sr(A):
    """ obtain the spectral radius of a square matrix A """
    return max(np.abs(np.linalg.eigvals(A)))

def companion_stack(Acoefs,verbose=False):
    """ put a list of Acoefs for lags into the companion form """
    num_lags = len(Acoefs)
    if num_lags == 1:
        if verbose:
            print('[companion stack] WARNING: num_lags = 1, degenerate companion form.')
        return Acoefs[0]
    p = Acoefs[0].shape[0]
    identity = np.diag(np.ones(((num_lags-1)*p,))) ## identity matrix of size (num_lags-1)*p
    zeros = np.zeros(((num_lags-1)*p,p))
    bottom = np.concatenate([identity, zeros],axis=1)
    top = np.concatenate(Acoefs,axis=1)
    return np.concatenate([top,bottom],axis=0)

def companion_disagg(mtx, num_lags, verbose=False):
    """ extract the lag coefficients from a companion form """
    if mtx.shape[1] == num_lags:
        if verbose:
            print('[companion disaggregate] nothing to disaggregate')
        return [mtx]
    Acoefs = []
    assert mtx.shape[1] % num_lags == 0, 'number of columns in the companion matrix is not a multiple of the number of lags; something went wrong'
    p = int(mtx.shape[1]//num_lags)
    for i in range(num_lags):
        Acoefs.append(mtx[:p,(i*p):((i+1)*p)])
    return Acoefs

def scale_coefs(Acoefs, target, verbose=False):
    """ scale the matrix so that its spectral radius is smaller than the target """
    num_lags = len(Acoefs)
    A_comp = companion_stack(Acoefs)
    old_sr = get_sr(A_comp)
    A_comp_scaled = target / old_sr * A_comp
    Acoefs_new = companion_disagg(A_comp_scaled, num_lags)
    ## since the scaling won't be exact, do some sanity checks
    A_comp_new = companion_stack(Acoefs_new)
    new_sr = get_sr(A_comp_new)
    if verbose:
        print(f'spectral radius before scaling = {old_sr:.3f}, after scaling = {new_sr:.3f}')
    if new_sr >= 1:
        print(f'![WARNING]: spectral radius after scaling = {new_sr:.3f}')
    return Acoefs_new

def gen_VAR1_coef(p, target_sr=0.5, scale_range=(0.5,1), diag = False):
    """
    generate the transition matrix for a stationary VAR(1) process
    Return: p*p matrix
    """
    raw_mtx = gen_randmtx(p, p, scale_range=scale_range, diag=diag)
    scaled_mtx = scale_coefs([raw_mtx],target=target_sr)[0]
    return scaled_mtx

def gen_VARd_coef(p, d=2, target_sr = 0.8, diag= False, scale_ranges = None):
    """
    generate the transition matrix for a stationary VAR(2) process
    Note that the target_sr will not be attained exactly
    Return: p*p*2 tensor
    """
    if not scale_ranges:
        scale_ranges = [[2*i,2*i+1] for i in range(d)]
        scale_ranges.reverse()
    raw_mtx_list= [gen_randmtx(p,p,scale_range=scale_ranges[i],diag=diag) for i in range(d)]
    scaled_mtx_list = scale_coefs(raw_mtx_list, target=target_sr)
    return np.stack(scaled_mtx_list,axis=-1)

def sigmoid(x):
    return 1./(1+np.exp(-x))

def rbfns(x1,x2,epsilon=1,kernel_type='gaussian'):
    d = np.sum((x1 - x2)**2)
    eps_sq = epsilon**2
    if kernel_type == 'gaussian':
        return np.exp(-eps_sq * d)
    elif kernel_type == 'multiquadratic':
        return (1 + eps_sq * d)**0.5
    elif kernel_type == 'inv_quadratic':
        return (1 + eps_sq * d)**(-1)
    elif kernel_type == 'inv_multiquadratic':
        return (1 + eps_sq * d)**(-0.5)
    else:
        raise ValueError(f"incorrect specification of kernel_type; choose among {['gaussian','multiquadratic','inv_quadratic','inv_multiquadratic']}")

def exp_almon_helper(delta,nlags):
    """raw coefficient for almon polynomial (before normalized)"""
    degree = len(delta)
    arr = np.zeros((degree,nlags))
    for s in range(degree):
        for j in range(nlags):
            arr[s,j] = delta[s] * (j**s)
    return np.exp(np.sum(arr,axis=0))

def gen_exp_almon(delta,nlags,normalize=True):
    coef = exp_almon_helper(delta,nlags)
    if normalize:
        return coef / np.sum(coef)
    else:
        return coef

def gen_exp_almon_mtx(p, q, nlags, normalize=True, deltas = None, degree = None, low=-1, high=1, sort = False, pm = False):
    """
    Argv:
        normalize: bool, whether the coefficients are normalized so that they sum up to 1
        deltas: p by q by *, where * is the list of coefficients for the polynomial, optional
        if deltas is None:
            degree, low, high - almon polynomial coefficient will be generated as np.random.uniform(low,high,size=(degree,))
            sort - bool, descending sort flag
            
    return: ret, p by q by nlags tensor, where ret[i,j,:] corresponds to the coefficient of lags for coordinate i (response), j (dependent)
    """
    
    ret = np.zeros((p,q,nlags))
    if deltas is not None:
        for i in range(p):
            for j in range(q):
                ret[i,j,:] = gen_exp_almon(deltas[i,j,:],nlags,normalize=normalize)
        return ret
        
    assert degree in [2,3]
    for i in range(p):
        for j in range(q):
            delta = np.random.uniform(low=low,high=high,size=(degree,))
            almon_coef = gen_exp_almon(delta,nlags,normalize=normalize)
            if sort:
                almon_coef = sorted(almon_coef,reverse=True)
            if pm:
                almon_coef *= np.random.choice([-1,1],size=len(almon_coef),replace=True)
            ret[i,j,:] = almon_coef
    
    return ret

def plot_expalmon_mtx(list_of_coords, mtx, savefig = ""):
    nlags = mtx.shape[-1]
    fig, ax = plt.subplots(1, 1,figsize=(7,4))
    x = np.arange(nlags)
    for i,j in list_of_coords:
        ax.plot(x,mtx[i-1,j-1,:],label=f'({i},{j})',marker='.')
    ax.legend(loc='best')
    if len(savefig):
        fig.savefig(savefig)

def export_data_to_excel(dict_of_data,filename):
    dim_info = {}
    dict_of_df = {}
    for key, data in dict_of_data.items():
        multiplier = 1 if key != 'y' else 3
        df = pd.DataFrame(data = data,
                          columns=[f'{key}{i}' for i in range(data.shape[1])],
                          index=[f't_{multiplier*i}' for i in range(data.shape[0])])
        df.index.name = 'timestamp'
        dict_of_df[key] = df
        dim_info.setdefault('variable',[]).append(f'dim_{key}')
        dim_info.setdefault('dim',[]).append(data.shape[1])
    dim = pd.DataFrame(dim_info)
    with pd.ExcelWriter(f'{filename}.xlsx') as writer:
        for key, df in dict_of_df.items():
            df.to_excel(writer,sheet_name=key,index=True)
        dim.to_excel(writer,sheet_name='dim_info',index=False)

def plot_data_rand_col(dict_of_data, filename):
    fig, axs = plt.subplots(len(dict_of_data),1,figsize=(12,6),constrained_layout=True)
    i = 0
    for key, data in dict_of_data.items():
        col_id = np.random.choice(list(range(data.shape[1])))
        if key in ['f','x']:
            axs[i].plot(np.arange(300), data[:300,col_id],marker='.')
        else:
            axs[i].plot(np.arange(100), data[:100,col_id],marker='.')
        axs[i].set_title(f'time series plot for {key}_{col_id}')
        i += 1
    fig.savefig(f'{filename}.png')
    plt.close('all')
            
