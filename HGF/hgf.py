""" Fuctions for model fitting and model simulation of the Hierarchical Gaussian Filter
code takes in stimulus states and simulates perception and prediction of an agent

Model implemented as discribed in: Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. Frontiers in human neuroscience, 8, 825.

Code adapted by Jorie van Haren (2021) """

# load nessecary packages
import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy import optimize

# load config files
from HGF.hgf_config import *

####################
## MAIN FUNCTIONS ##
####################

def hgf_binary(r, p, trans=False):
    """calculate trajectorie of agent's representations under HGF"""
    
    if trans: p = r['c_prc']['transp_prc_fun'](r, p) # transform parameters to native space
    p_dict = _unpack_para(p, r)            # get parameters unpacked
    u = np.insert(r['u'], 0, 0)            # add zeroth trial
    n = len(u)                             # length of trials inc. prior
    l = r['c_prc']['n_levels']             # get number of levels
    
    # set time dim for irregular intervals, or set to ones for reggular
    if r['c_prc']['irregular_intervals']:
        t = r['u'][1,:]  # make sure this deminsion is [2, x] second being time
    else:
        t = np.ones(n)
    
    # initialize what to update
    mu = np.empty((n, l)) * np.nan         # mu represnetation
    pi = np.empty((n, l)) * np.nan         # pi representation
    mu_hat = np.empty((n, l)) * np.nan     # mu^ quantity
    pi_hat = np.empty((n, l)) * np.nan     # pi^ quantity
    v = np.empty((n, l)) * np.nan           
    w = np.empty((n, l-1)) * np.nan
    da = np.empty((n, l)) * np.nan         # prediction errors
    
    # initial priors, for all remaining this will remain nan
    mu[0,0] = _sgm(p_dict['mu_0'][0], 1)
    mu[0,1:] = p_dict['mu_0'][1:]
    pi[0,0] = np.inf
    pi[0,1:] = p_dict['sa_0'][1:]**-1   # silence warning, inf resulst for sim model is fine
    
    # represnetation update loop!
    for trial in range(1, n):
        
        # check if trail has to be ignored
        if not trial in r['ign']:

            # make second level initial pred. (weighted by time)
            mu_hat[trial,1] = mu[trial-1,1] + (t[trial]*p_dict['rho'][1])     

            ####1ST LVL####
            # make first level pred using second level pred. 
            mu_hat[trial,0] = _sgm(p_dict['ka'][0] * mu_hat[trial,1], 1)  # prediction
            pi_hat[trial,0] = 1 / (mu_hat[trial,0] * (1-mu_hat[trial,0])) # precision of pred

            # update
            pi[trial,0] = np.inf
            mu[trial,0] = u[trial]

            # prediction error
            da[trial,0] = mu[trial,0] - mu_hat[trial,0]
            
            ####LOOP OVER LEVELS - TAKING SPECIAL CARE OF 2ND AND LAST LEVEL####
            for lvl in range(1, l):

                # for level 2
                if lvl < 2:
                    
                    # precision of prediction
                    pi_hat[trial,lvl] = (pi[trial-1,lvl]**-1 + t[trial] 
                                         * np.exp(p_dict['ka'][lvl] * 
                                                  mu[trial-1, lvl+1] +
                                                  p_dict['om'][lvl]))**-1

                    # update
                    pi[trial,1] = pi_hat[trial,1] + p_dict['ka'][0]**2 / pi_hat[trial,0]
                    mu[trial,1] = mu_hat[trial,1] + p_dict['ka'][0] / pi[trial,1] * da[trial,0]

                elif lvl >= 2: # all higher levels, scales above 3

                    # prediction (identical to initial pred)
                    mu_hat[trial,lvl] = mu[trial-1,lvl] + (t[trial]*p_dict['rho'][lvl])
                    
                    if lvl == l-1: # for last level
                        
                        # precision of prediction (now using -th-)
                        pi_hat[trial,l-1] = (pi[trial-1,l-1]**-1 + t[trial] * p_dict['th'])**-1
                        
                        # weighting factor
                        v[trial,l-1] = t[trial] * p_dict['th']
                        v[trial,l-2] = t[trial] * np.exp(p_dict['ka'][l-2] * mu[trial-1, l-1] + p_dict['om'][l-2])
                        w[trial,l-2] = v[trial,l-2] * pi_hat[trial,l-2]

                    else: # intermediate (not last/first) levels
                        
                        # precision of prediction (now using -th-)
                        pi_hat[trial,l-1] = (pi[trial-1,l-1]**-1 + t[trial] * p_dict['th'])**-1
                        
                        # weighting
                        v[trial, lvl-1] = t[trial] * np.exp(p_dict['ka'][lvl-1] * 
                                                            mu[trial-1, lvl] + 
                                                            p_dict['om'][lvl-1])
                        w[trial, lvl-1] = v[trial, lvl-1] * pi_hat[trial, lvl-1]

                        
                    ##---------------------------------------------------------------------------------------------------------##   
                    # updates using enhanced hgf binary model
                    if 'ehgf' in r['c_prc']['model']:
                        mu[trial,lvl] = mu_hat[trial,lvl] + \
                                        0.5 * pi_hat[trial,lvl]**-1 * \
                                        p_dict['ka'][lvl-1] * \
                                        w[trial,lvl-1] * \
                                        da[trial,lvl-1]
                        # update precision depending on mean update
                        vv = t[trial] * np.exp(p_dict['ka'][lvl-1]* 
                                               mu[trial, lvl] + 
                                               p_dict['om'][lvl-1])
                        pim_hat = (pi[trial-1, lvl-1]**-1 + vv)**-1
                        ww = vv * pim_hat
                        rr = (vv - pi[trial-1, lvl-1]**-1) * pim_hat
                        dd = (pi[trial, lvl-1]**-1 + (mu[trial, lvl-1] - mu_hat[trial, lvl-1])**2) * pim_hat -1
                        # update pi, add 0 if equation is lower then 0
                        pi[trial, lvl] = pi_hat[trial, lvl] + np.maximum(0, 0.5* p_dict['ka'][lvl-1]**2 * ww * (ww + rr * dd))
                    # we default back to standard hgf
                    else:
                        pi[trial,lvl] = pi_hat[trial,lvl] + \
                                        0.5 * p_dict['ka'][lvl-1]**2 * \
                                        w[trial,lvl-1] * \
                                        (w[trial,lvl-1] + (2 *w[trial,lvl-1] -1) *da[trial,lvl-1])
                        mu[trial,lvl] = mu_hat[trial,lvl] + \
                                        0.5 * pi[trial,lvl]**-1 * \
                                        p_dict['ka'][lvl-1] * \
                                        w[trial,lvl-1] * \
                                        da[trial,lvl-1]
                    ##---------------------------------------------------------------------------------------------------------## 
                        
                # prediction error    
                da[trial,lvl] = (pi[trial,lvl]**-1 + (mu[trial,lvl] - mu_hat[trial, lvl])**2)  *  pi_hat[trial,lvl] -1

        # if trial is ignored we do not update anything
        else: 
            mu[trial,:] = mu[trial-1,:]
            pi[trial,:] = pi[trial-1,:]

            mu[trial,:] = mu[trial-1,:]
            pi[trial,:] = pi[trial-1,:]

            v[trial,:]  = v[trial-1,:]
            w[trial,:]  = w[trial-1,:]
            da[trial,:] = da[trial-1,:]
    
    # learing rates
    sgmmu2    = _sgm(p_dict['ka'][0] * mu[:,1], 1)
    dasgmmu2  = u - sgmmu2   
    lr1       = np.divide(np.diff(sgmmu2), dasgmmu2[1:n])
    lr1[da[1:n,1]==0] = 0
    
    # remove rep. priors and dummy value
    mu       = np.delete(mu,0, axis=0)
    pi       = np.delete(pi,0, axis=0)
    mu_hat   = np.delete(mu_hat,0, axis=0)
    pi_hat   = np.delete(pi_hat,0, axis=0)
    v        = np.delete(v,0, axis=0)
    w        = np.delete(w,0, axis=0)
    da       = np.delete(da,0, axis=0)
    
    # store results in dict
    traj = {}
    traj['mu']      = mu
    traj['sa']      = pi**-1
    traj['mu_hat']  = mu_hat
    traj['sa_hat']  = pi_hat**-1
    traj['v']       = v
    traj['w']       = w
    traj['da']      = da
    traj['ud']      = mu - mu_hat  # updates with respect to prediction
    
    # precision weight on pred error
    psi          = np.empty([n-1,l])
    psi[:]       = np.nan
    psi[:,1]     = pi[:,1]**-1
    psi[:,2:l]   = np.divide(pi_hat[:,1:l-1], pi[:,2:l])
    traj['psi']  = psi
    
    # epsions (precision weighted pred. errors)
    epsi         = np.empty([n-1,l])
    epsi[:]      = np.nan
    epsi[:,1:l]  = np.multiply(psi[:,1:l], da[:,:l-1])
    traj['epsi'] = epsi
    
    # learning rate
    wt           = np.empty([n-1,l])
    wt[:]        = np.nan    
    wt[:,0]      = lr1
    wt[:,1]      = psi[:,1]
    wt[:,2:l]    = np.multiply(0.5 * (v[:,1:l-1] * 
                                      np.diagonal(p_dict['ka'][1:l-1].reshape(1,len(p_dict['ka'][1:l-1])))), 
                                      psi[:,2:l])
    traj['wt']   = wt
    
    # matrics observational model DIMENSIONALLITY PROBLEMS
    infStates    = np.empty([n-1,l,4])
    infStates[:] = np.nan
    infStates[:,:,0]  = traj['mu_hat']
    infStates[:,:,1]  = traj['sa_hat']
    infStates[:,:,2]  = traj['mu']
    infStates[:,:,3]  = traj['sa']
    return([traj, infStates])

def ehgf_binary(r, p, trans=False):
    """Allias function for hgf_binary with r['c_prc']['model'] set to 'ehgf_binary'"""
    # set model manually to ehgf_binary for enhanced model
    r['c_prc']['model'] = 'ehgf_binary'
    return(hgf_binary(r, p, trans=trans))



def hgf(r, p, trans=False):
    """calculate trajectorie of agent's representations under HGF"""
    
    if trans: p = r['c_prc']['transp_prc_fun'](r, p) # transform parameters to native space
    p_dict = _unpack_para(p, r)            # get parameters unpacked
    u = np.insert(r['u'], 0, 0)            # add zeroth trial
    n = len(u)                             # length of trials inc. prior
    l = r['c_prc']['n_levels']             # get number of levels
    
    # set time dim for irregular intervals, or set to ones for reggular
    if r['c_prc']['irregular_intervals']:
        t = r['u'][1,:]  # make sure this deminsion is [2, x] second being time
    else:
        t = np.ones(n)
    
    # initialize what to update
    mu = np.empty((n, l)) * np.nan         # mu represnetation
    pi = np.empty((n, l)) * np.nan         # pi representation
    mu_hat = np.empty((n, l)) * np.nan     # mu^ quantity
    pi_hat = np.empty((n, l)) * np.nan     # pi^ quantity
    v = np.empty((n, l)) * np.nan           
    w = np.empty((n, l-1)) * np.nan
    da = np.empty((n, l)) * np.nan         # prediction errors
    dau = np.empty((n, 1)) * np.nan
    
    # initial priors, for all remaining this will remain nan
    mu[0,:] = p_dict['mu_0']
    pi[0,:] = p_dict['sa_0']**-1
    
    # represnetation update loop!
    for trial in range(1, n):
        
        # check if trail has to be ignored
        if not trial in r['ign']:  

            ####1ST LVL####
            # make first level pred, and precision of prediction
            mu_hat[trial,0] = mu[trial-1, 0] + t[trial] * p_dict['rho'][0]
            pi_hat[trial,0] = (pi[trial-1, 0]**-1 + t[trial] * np.exp(p_dict['ka'][0] *
                                                                     mu[trial-1, 1] + 
                                                                     p_dict['om'][0]))**-1
            
            # pred. error input
            dau[trial] = u[trial] - mu_hat[trial, 0]

            # update
            pi[trial,0] = pi_hat[trial, 0] + p_dict['al']**-1
            mu[trial,0] = mu_hat[trial, 0] + pi_hat[trial, 0]**-1 * \
                          (pi_hat[trial, 0]**-1 + p_dict['al'])**-1 * \
                          dau[trial]
            
            # volatility prediction error
            da[trial,0] = (pi[trial,0]**-1 + (mu[trial,0] - mu_hat[trial,0])**2) * \
                          pi_hat[trial,0] - 1
            
            ####LOOP OVER LEVELS - TAKING SPECIAL CARE OF 2ND AND LAST LEVEL####
            for lvl in range(1, l):

                # prediction (identical to initial pred)
                mu_hat[trial,lvl] = mu[trial-1,lvl] + (t[trial]*p_dict['rho'][lvl])

                if lvl != l-1: # for last level
                    # precision of prediction
                    pi_hat[trial,lvl] = (pi[trial-1,lvl]**-1 + t[trial] 
                                         * np.exp(p_dict['ka'][lvl] * 
                                                  mu[trial-1, lvl+1] +
                                                  p_dict['om'][lvl]))**-1
                    
                    # weighting
                    v[trial, lvl-1] = t[trial] * np.exp(p_dict['ka'][lvl-1] * 
                                                        mu[trial-1, lvl] + 
                                                        p_dict['om'][lvl-1])
                    w[trial, lvl-1] = v[trial, lvl-1] * pi_hat[trial, lvl-1]

                else: # intermediate (not last/first) levels
                   # precision of prediction (now using -th-)
                    pi_hat[trial,l-1] = (pi[trial-1,l-1]**-1 + t[trial] * p_dict['th'])**-1  

                    # weighting factor
                    v[trial,l-1] = t[trial] * p_dict['th']
                    v[trial,l-2] = t[trial] * np.exp(p_dict['ka'][l-2] * mu[trial-1, l-1] + p_dict['om'][l-2])
                    w[trial,l-2] = v[trial,l-2] * pi_hat[trial,l-2]
                    

                ##---------------------------------------------------------------------------------------------------------##    
                # UPDATES USING ENCHANCED HGF MODEL
                if 'ehgf' in r['c_prc']['model']:
                    mu[trial,lvl] = mu_hat[trial,lvl] + \
                                    0.5 * pi_hat[trial,lvl]**-1 * \
                                    p_dict['ka'][lvl-1] * \
                                    w[trial,lvl-1] * \
                                    da[trial,lvl-1]
                    # update precision depending on mean update
                    vv = t[trial] * np.exp(p_dict['ka'][lvl-1]* 
                                           mu[trial, lvl] + 
                                           p_dict['om'][lvl-1])
                    pim_hat = (pi[trial-1, lvl-1]**-1 + vv)**-1
                    ww = vv * pim_hat
                    rr = (vv - pi[trial-1, lvl-1]**-1) * pim_hat
                    dd = (pi[trial, lvl-1]**-1 + (mu[trial, lvl-1] - mu_hat[trial, lvl-1])**2) * pim_hat -1
                    # update pi, add 0 if equation is lower then 0
                    pi[trial, lvl] = pi_hat[trial, lvl] + np.maximum(0, 0.5* p_dict['ka'][lvl-1]**2 * ww * (ww + rr * dd))
                    
                # OR WE DEFAULT TO STANDARD HGF MODEL
                else:
                    pi[trial,lvl] = pi_hat[trial,lvl] + \
                                    0.5 * p_dict['ka'][lvl-1]**2 * \
                                    w[trial,lvl-1] * \
                                    (w[trial,lvl-1] + (2 *w[trial,lvl-1] -1) *da[trial,lvl-1])
                    mu[trial,lvl] = mu_hat[trial,lvl] + \
                                    0.5 * pi[trial,lvl]**-1 * \
                                    p_dict['ka'][lvl-1] * \
                                    w[trial,lvl-1] * \
                                    da[trial,lvl-1]
                ##---------------------------------------------------------------------------------------------------------## 
                        
                # prediction error    
                da[trial,lvl] = (pi[trial,lvl]**-1 + (mu[trial,lvl] - mu_hat[trial, lvl])**2)  *  pi_hat[trial,lvl] -1

        # if trial is ignored we do not update anything
        else: 
            mu[trial,:] = mu[trial-1,:]
            pi[trial,:] = pi[trial-1,:]

            mu[trial,:] = mu[trial-1,:]
            pi[trial,:] = pi[trial-1,:]

            v[trial,:]  = v[trial-1,:]
            w[trial,:]  = w[trial-1,:]
            da[trial,:] = da[trial-1,:]
    
    # remove rep. priors and dummy value
    mu       = np.delete(mu,0, axis=0)
    pi       = np.delete(pi,0, axis=0)
    mu_hat   = np.delete(mu_hat,0, axis=0)
    pi_hat   = np.delete(pi_hat,0, axis=0)
    v        = np.delete(v,0, axis=0)
    w        = np.delete(w,0, axis=0)
    da       = np.delete(da,0, axis=0)
    dau      = np.delete(dau, 0)
    
    # store results in dict
    traj = {}
    traj['mu']      = mu
    traj['sa']      = pi**-1
    traj['mu_hat']  = mu_hat
    traj['sa_hat']  = pi_hat**-1
    traj['v']       = v
    traj['w']       = w
    traj['da']      = da
    traj['dau']     = dau.reshape(len(dau),1)
    traj['ud']      = mu - mu_hat  # updates with respect to prediction
    
    # precision weight on pred error
    psi          = np.empty([n-1,l])
    psi[:]       = np.nan
    psi[:,0]     = (p_dict['al'] * pi[:,0])**-1
    psi[:,1:l]   = np.divide(pi_hat[:,0:l-1], pi[:,1:l])
    traj['psi']  = psi
    
    # epsions (precision weighted pred. errors)
    epsi         = np.empty([n-1,l])
    epsi[:]      = np.nan
    epsi[:,0]    = np.multiply(psi[:,0], dau)
    epsi[:,1:l]  = np.multiply(psi[:,1:l], da[:,:l-1])
    traj['epsi'] = epsi
    
    # learning rate
    wt           = np.empty([n-1,l])
    wt[:]        = np.nan    
    wt[:,0]      = psi[:,0]
    wt[:,1:l]    = np.multiply(0.5 * (v[:,0:l-1] * 
                                      np.diagonal(p_dict['ka'][0:l-1].reshape(1,len(p_dict['ka'][0:l-1])))), 
                                      psi[:,1:l])
    traj['wt']   = wt
    
    # matrics observational model DIMENSIONALLITY PROBLEMS
    infStates    = np.empty([n-1,l,4])
    infStates[:] = np.nan
    infStates[:,:,0]  = traj['mu_hat']
    infStates[:,:,1]  = traj['sa_hat']
    infStates[:,:,2]  = traj['mu']
    infStates[:,:,3]  = traj['sa']
    return([traj, infStates])

def ehgf(r, p, trans=False):
    """Allias function for hgf with r['c_prc']['model'] set to 'ehgf'"""
    # set model manually to ehgf_binary for enhanced model
    r['c_prc']['model'] = 'ehgf'
    return(hgf(r, p, trans=trans))


## Transform parameters

def hgf_transp(r, ptrans):
    """transform parameters to native space"""
    # initialize nan array
    pvec = np.empty(len(ptrans))
    pvec[:] = np.nan
    
    # get number of levels
    l = r['c_prc']['n_levels']
    
    # trans to native space
    pvec[0:l]          = ptrans[0:l]
    pvec[l:2*l]        = np.exp(ptrans[l:2*l])
    pvec[2*l:3*l]      = ptrans[2*l:3*l]
    pvec[3*l:4*l-1]    = np.exp(ptrans[3*l:4*l-1])
    pvec[4*l-1:5*l-1]  = ptrans[4*l-1:5*l-1]
    # for continuus hgf
    if not 'binary' in r['c_prc']['model']:
        pvec[5*l-1]    = np.exp(ptrans[5*l-1])
    return(pvec)

def unitsq_sqm_transp(r, ptrans):
    """transform parameters to native space"""
    # initialize nan array
    pvec = np.empty(len(ptrans))
    pvec[:] = np.nan
    pstruct = {}
    
    # get _ze_
    pvec[0] = np.exp(ptrans)
    pstruct['ze'] = pvec[0]
    return([pvec, pstruct])


## Calculations optimization

def bayes_optimal_binary(r, infStates, ptrans):
    """calculate the log-probabilitie of inputs given predictions"""
    # initialize arrays
    n        = infStates.shape[0]
    logp     = np.empty(n)
    logp[:]  = np.nan
    y_hat    = np.empty(n)
    y_hat[:] = np.nan
    res      = np.empty(n)
    res[:]   = np.nan
    
    # remove irregulars 
    u = r['u'][:]                
    u = np.delete(u, r['irr'])     # for inputs
    x = infStates[:,0,0]
    x = np.delete(x, r['irr'])     # and for predictions

    # calculate log-prob for remaining trials
    reg       = ~np.isin(np.arange(0, len(u)), r['irr'])
    logp[reg] = np.multiply(u, np.log(x)) + np.multiply(1-u, np.log(1-x))
    y_hat[reg] = x
    res[reg]  = np.divide(u-x, np.sqrt(np.multiply(x, 1-x)))
    return(logp, y_hat, res)


def bayes_optimal(r, infStates, ptrans):
    """calculate the log-probabilitie of inputs given predictions"""
    # initialize arrays
    n        = infStates.shape[0]
    logp     = np.empty(n)
    logp[:]  = np.nan
    y_hat    = np.empty(n)
    y_hat[:] = np.nan
    res      = np.empty(n)
    res[:]   = np.nan
    
    # remove irregulars 
    u = r['u'][:]                
    u = np.delete(u, r['irr'])     # for inputs

    # predictions
    mu1hat = infStates[:,0,0]
    mu1hat = np.delete(mu1hat, r['irr'])
    
    # variance {inverse precision} of prediction
    sa1hat = infStates[:,0,1]
    sa1hat = np.delete(sa1hat, r['irr'])
    
    # calculate log-prob for remaining trials
    reg        = ~np.isin(np.arange(0, len(u)), r['irr'])
    logp[reg]  = -0.5 * np.log((8*np.arctan(1)) * sa1hat) - \
                        np.divide((u - mu1hat)**2, 2*sa1hat)
    y_hat[reg] = mu1hat
    res[reg]   = u-mu1hat
    return(logp, y_hat, res)


## calculations for observational models

def gaussian_obs(r, infStates, ptrans):
    """Calculate log-probabilities of y=1 using gaussian noise model"""
    # initialize arrays
    n        = infStates.shape[0]
    logp     = np.empty(n)
    logp[:]  = np.nan
    y_hat    = np.empty(n)
    y_hat[:] = np.nan
    res      = np.empty(n)
    res[:]   = np.nan
    
    # remove irregulars 
    u = r['u'][:]                
    u = np.delete(u, r['irr'])     # for inputs
    
    # zeta to native
    ze = np.exp(ptrans[0])
    
    # remove irregulars
    x = infStates[:,0,0]
    x = np.delete(x, r['irr'])     # and for predictions    
    y = r['y'][:]                
    y = np.delete(y, r['irr'])     # for perception
  
    # calculate log-prob for remaining trials
    reg        = ~np.isin(np.arange(0, len(u)), r['irr']) 
    logp[reg]  = -0.5 * np.log((8*np.arctan(1)) * ze) - \
                        np.divide((y - x)**2, 2*ze)
    y_hat[reg] = x
    res[reg]   = y-x
    
    return(logp, y_hat, res)


def unitsq_sgm(r, infStates, ptrans):
    """Calculate log-probabilities of y=1 using unit-sq sigmoid model"""
    # initialize arrays
    n        = infStates.shape[0]
    logp     = np.empty(n)
    logp[:]  = np.nan
    y_hat    = np.empty(n)
    y_hat[:] = np.nan
    res      = np.empty(n)
    res[:]   = np.nan
    
    # remove irregulars 
    u = r['u'][:]                
    u = np.delete(u, r['irr'])     # for inputs
    
    # zeta to native
    ze = np.exp(ptrans[0])
    
    # remove irregulars
    x = infStates[:,0,0]
    x = np.delete(x, r['irr'])     # and for predictions    
    y = r['y'][:]                
    y = np.delete(y, r['irr'])     # for perception
    
    # logtransform 
    logx                = np.log(x)
    logx[1-x < 1e-4]    = np.log1p(x-1)[1-x < 1e-4]   # so we dont get any rounding errors later on
    logminx             = np.log(1-x)
    logminx[x < 1e-4]   = np.log1p(-x)[x < 1e-4]      # so we dont get any rounding errors later on
  
    # calculate log-prob for remaining trials
    reg        = ~np.isin(np.arange(0, len(u)), r['irr']) 
    logp[reg]  = np.multiply(np.multiply(y, ze),
                             logx - logminx) + np.multiply(ze, logminx) - np.log((1-x)**ze + x**ze)
    y_hat[reg] = x
    res[reg]   = np.divide(y-x,
                           np.sqrt(np.multiply(x,
                                               1-x)))
    
    return(logp, y_hat, res)

## helper functions

def _unpack_para(p, r):
    """inside function, not to be called from outside
    takes in parameters and unpack them"""
    # get number of levels
    l = r['c_prc']['n_levels']
    
    # unpack parameters into dict
    p_dict = {}
    p_dict['mu_0']    = p[0:l]
    p_dict['sa_0']    = p[l:2*l]
    p_dict['rho']     = p[2*l:3*l]
    p_dict['ka']      = p[3*l:4*l-1]
    p_dict['om']      = p[4*l-1:5*l-2]
    with np.errstate(divide='ignore'): p_dict['th'] = np.exp(p[5*l-2])
    # for continuus hgf
    if not 'binary' in r['c_prc']['model']:
        p_dict['pi_u']  = p[5*l-1]
        p_dict['al']    = 1/p[5*l-1]
    return(p_dict)


def _sgm(x, a):
    return(np.divide(a,1+np.exp(-x)))
