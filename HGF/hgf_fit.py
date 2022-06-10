""" Fuctions for model fitting and model simulation of the Hierarchical Gaussian Filter
code takes in stimulus states and simulates perception and prediction of an agent

Model implemented as discribed in: Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. Frontiers in human neuroscience, 8, 825.

Code adapted by Jorie van Haren (2021) """

# load nessecary packages
import numpy as np
import sys
import statsmodels.api as sm
from scipy import optimize

# load config files and hgf update functions
from HGF.hgf_config import *
from HGF.hgf import *
from HGF.hgf_pres import *

# load extra (non exclusive) helper function
from HGF.hgf import _unpack_para

#######################
## MAIN FIT FUNCTION ##
#######################

def fitModel(responses, inputs, 
             per_model=ehgf_binary_config, 
             obs_model=unitsq_sgm_config, 
             opt_model=quasinewton_optim_config,
             overwrite_opt=False):
    """Main function for fitting parameters of perceptual and obserrvational models
    input:  
            responses =  list or array of binary responses
            inputs    =  list or array of inputs
    optional inputs: 
            per_model  =  what perceptual model (see functions) to use
            obs_model  =  what observational model (see functions) to use
            opt_model  =  what optimization model to use (so far only implemented quasinewton_optim)
            
            overwrite_opt = default False, or a dictionary with personal settings instead of 
                            retrieving them from _config
                         - Dict should have dict['c_prc'], dict['c_obs'], and/or dict['c_opt']
                         - In here you may place keys with own options
                         - e.g. overwrite_optr['c_prc']['rhomu'] = np.array(['np.nan, 0.5, 0.5'])
    output:
            returns a dict r with inputs, outputs optimizations trajactories and all settings
    """
    
    # initialize r dict
    r = _dataPrep(responses, inputs)

    # set models
    r['c_prc'] = per_model()  # set perceptual model    
    r['c_prc']['config'] = per_model
    r['c_obs'] = obs_model()  # set observation model
    r['c_obs']['config'] = obs_model
    r['c_opt'] = opt_model()  # set optimization algoritm
    r['c_opt']['config'] = opt_model

    # override with our own settings
    if overwrite_opt != False:
        for item in ['c_prc', 'c_obs', 'c_opt']:
            if item not in overwrite_opt: overwrite_opt[item] = {}
            r[item] = {**r[item], **overwrite_opt[item]}

    # replace placeholder parameters with calculated values
    r['c_prc']['priormus'] = np.array([r['plh']['p99991'] if i == 99991 else i for i in r['c_prc']['priormus']])
    r['c_prc']['priorsas'] = np.array([r['plh']['p99991'] if i == 99991 else i for i in r['c_prc']['priorsas']])

    r['c_prc']['priormus'] = np.array([r['plh']['p99992'] if i == 99992 else i for i in r['c_prc']['priormus']])
    r['c_prc']['priorsas'] = np.array([r['plh']['p99992'] if i == 99992 else i for i in r['c_prc']['priorsas']])

    r['c_prc']['priormus'] = np.array([r['plh']['p99993'] if i == 99993 else i for i in r['c_prc']['priormus']])
    r['c_prc']['priorsas'] = np.array([r['plh']['p99993'] if i == 99993 else i for i in r['c_prc']['priorsas']])

    r['c_prc']['priormus'] = np.array([-r['plh']['p99993'] if i == -99993 else i for i in r['c_prc']['priormus']])
    r['c_prc']['priorsas'] = np.array([-r['plh']['p99993'] if i == -99993 else i for i in r['c_prc']['priorsas']])

    r['c_prc']['priormus'] = np.array([r['plh']['p99994'] if i == 99994 else i for i in r['c_prc']['priormus']])
    r['c_prc']['priorsas'] = np.array([r['plh']['p99994'] if i == 99994 else i for i in r['c_prc']['priorsas']])

    r.pop('plh') # clean up dict

    # estimate mode of posterior parameter distr. (M.A.P. estimate)
    with np.errstate(divide='ignore'):
        r = _optim(r, r['c_prc']['prc_fun'], r['c_obs']['obs_fun'], r['c_opt']['opt_fun'])

    # get perceptual and observation parameters
    n_prcpars = len(r['c_prc']['priormus'])
    ptrans_prc = r['optim']['final'][:n_prcpars]
    ptrans_obs = r['optim']['final'][n_prcpars:]

    # transform parameters back into dict percetpual model
    r['p_prc']         = _unpack_para(r['c_prc']['transp_prc_fun'](r, ptrans_prc), r)
    r['p_prc']['p']    = r['c_prc']['transp_prc_fun'](r, ptrans_prc)
    r['p_prc']['ptrans'] = ptrans_prc

    # transform parameters back into dict observational model
    if np.any(ptrans_obs): 
        _, r['p_obs']       = r['c_obs']['transp_obs_fun'](r, ptrans_obs) # _unpack_para(XXX , r)
        r['p_obs']['p'], _, = r['c_obs']['transp_obs_fun'](r, ptrans_obs)
    else: 
        r['p_obs']      = {}
        r['p_obs']['p'] = []
    r['p_obs']['ptrans'] = ptrans_obs

    # store estimates, predictions and risiduals
    with np.errstate(divide='ignore'): r['traj'], infStates  = r['c_prc']['prc_fun'](r, r['p_prc']['ptrans'], trans=True)  # ignore /0 warning here, since it will correctly give inf.
    _, r['optim']['yhat'], r['optim']['res'] = r['c_obs']['obs_fun'](r, infStates, r['p_obs']['ptrans'])

    # autocorrelation of risiduals
    res = r['optim']['res']
    res = np.nan_to_num(res)  # for irregular trials
    r['optim']['resAC'] = sm.tsa.acf(res, nlags=res.size, fft=True)

    # display results
    printfitmodel(r)
    return(r)


## Helper functions

def _dataPrep(responses, inputs):
    """internal function, not to be used from outside
    function stores responses, input and info in new dictonary r
    it also defines defaults, values within this dictonary that can later be overwritten
    input: responses
    output: dictonary r with placeholder info and y/u/irragular/ignored values"""
    
    # we first initiate a data dict
    r = {}
    
    # store responses and inputs
    r['y'] = np.array(responses)
    r['u'] = np.array(inputs)
    
    # next inputs if we dont have time axis
    if inputs.ndim == 1: r['u'].reshape(1,len(inputs))
    
    # check for ignored trials and irregular trials
    r['ign'] = np.argwhere(np.isnan(r['u']))
    r['irr'] = np.argwhere(np.isnan(r['y']))
    
    # display both ignored and irregular trials
    print('Ignored trials: {}'.format(r['ign']))
    print('Irregular trials: {}'.format(r['irr']))
    
    ## set placeholder values
    r['plh'] = {}                                 # nested dictionary for storing config files
    r['plh']['p99991'] = r['u'][0]                # set prior mean of mu_1
    r['plh']['p99992'] = np.var(r['u'][:20])      # set prior variance of mu_1 (using first 20 inputs/less if size is limited)
    r['plh']['p99993'] = np.log(r['plh']['p99992'])    # set prior mean log(sa_1) and alpha using log-var of first 20
    r['plh']['p99994'] = np.log(r['plh']['p99992']) -2 # setprior mean of emega_1 using first 20 log var - 2
    return(r)

def _negLogJoint(r, prc_fun, obs_fun, ptrans_prc, ptrans_obs):
    """returns the negative log-joint density for 
    perceptual and observational parameters"""
    # calc. perceptual trajectories, 
    [dummy, infStates] = prc_fun(r, ptrans_prc, trans=True)
    
    # calc. log-likelihood of observed responses given perceptual trajectories
    trialLogLls, y_hats, res = obs_fun(r, infStates, ptrans_obs)
    logLl = np.nansum(trialLogLls)
    negLogLl = -logLl
    
    # calc. log-prior of perceptual parameters
    prc_idx = r['c_prc']['priorsas']
    prc_idx = np.argwhere(~np.isnan(prc_idx) & (prc_idx > 0))
    logPrcPriors = _calclogpriors(r['c_prc'], ptrans_prc, prc_idx)
    logPrcPrior  = np.sum(logPrcPriors)                           
    
    # calc. log-prior of observation parameters
    obs_idx = r['c_obs']['priorsas']
    obs_idx = np.argwhere(~np.isnan(obs_idx) & (obs_idx > 0))
    logObsPriors = _calclogpriors(r['c_obs'], ptrans_obs, obs_idx)
    logObsPrior  = np.sum(logObsPriors)    
    
    # concatenate calculations
    negLogJoint = -(logLl + logPrcPrior + logObsPrior)
    return(negLogJoint, negLogLl)

def _optim(r, prc_fun, obs_fun, opt_fun):
    """internal function, not to be used from outside
    function determines parameters to optimize and does optimalization run(s)
    it records these optimiziation results"""
    # sellect parameters that are not fixed or NaN
    opt_idx = np.array(r['c_prc']['priorsas'].tolist() + r['c_obs']['priorsas'].tolist())
    opt_idx = np.nonzero([0 if np.isnan(i) else i for i in opt_idx])[0]
    
    # set perceptual and observation par lengths
    n_prcpars = len(r['c_prc']['priormus'])
    n_obspars = len(r['c_obs']['priormus'])
    
    # construct objective function to be minimized (var to be minimized p)
    nlj = lambda p: _negLogJoint(r, prc_fun, obs_fun, p[0:n_prcpars], p[n_prcpars:n_prcpars+n_obspars])

    # initiate by setting the prior mean as starting value for optimization
    init = np.array(r['c_prc']['priormus'].tolist() + r['c_obs']['priormus'].tolist())
    dummy1, dummy2= nlj(init)  # check could be error: last p in
    
    # do an optimization run and record opt. results
    optres = _optimrun(nlj, init, opt_idx, r['c_opt']['config'], r['c_opt'])
    optres['init']  = np.array(r['c_prc']['priormus'].tolist() + r['c_obs']['priormus'].tolist())
    
    # record opt results
    r['optim'] = {}
    for key in optres:
        r['optim'][key] = optres[key]
    
    # calc AIC/BIC
    d = len(opt_idx)
    if np.any(r['y']):
        ndp = np.nansum(r['y'])
    else:
        ndp = np.nansum(r['u'])
    
    r['optim']['AIC'] = 2*r['optim']['negLl']  +  2*d
    r['optim']['BIC'] = 2*r['optim']['negLl']  +  2*np.log(ndp)
    return(r)


def _optimrun(nlj, init, opt_idx, opt_fun, c_opt):
    """internal function not to be called from outside
    does an (1) optimization algorithm run and returns results"""
    
    # objective function with respect to parameters that are not optimized
    obj_fun = lambda p_opt: _restrictfun(nlj, init, opt_idx, p_opt)
    
    # optimize
    print("\nInitializing optimization run...\n") 
    optresz = c_opt['opt_fun'](obj_fun, init[opt_idx], 
                               method=c_opt['opt_method'],
                               options={'return_all':True,
                               'gtol':c_opt['tolGrad'],
                               'maxiter':c_opt['maxIter'],
                               'disp':True})
    
    optres = {}
    optres['valMin']  = optresz['fun'] 
    optres['argMin']  = optresz['x']
#     optres['init']    = init_og
    final             = init
    final[opt_idx]    = optres['argMin']
    optres['final']   = final
    
    # get neg log-joint and log likelihood
    negLj, negLl = nlj(final)
    d = len(opt_idx)
    
    # computation of hessian
    optres['H']       = _get_near_psd(np.linalg.inv(optresz['hess_inv']))
    optres['Sigma']   = _get_near_psd(optresz['hess_inv'])
    optres['Corr']    = _correlation_from_covariance(optres['Sigma'])
    optres['negLl']   = negLl
    optres['negLj']   = negLj
    optres['LME']     = -optres['valMin'] + 0.5*np.log(np.linalg.det(optres['H'])**-1) + d/(2*np.log(2*np.pi))
    optres['accu']    = -negLl
    optres['comp']    = optres['accu'] - optres['LME']

    # return dict
    return(optres)


def _calclogpriors(r, ptrans, idx):
    """internal function not to be called from outside
    returns log-priors of parameters - perceptual or observational"""
    
    # check if array is empty
    if idx.size != 0:  
        
        # calculate log piors
        logPrior = np.multiply(-.5, 
                               np.log(np.multiply(8*np.arctan(1),
                                                  r['priorsas'][idx]))) - \
                   np.divide(np.multiply(.5, ptrans[idx] - r['priormus'][idx])**2,
                             r['priorsas'][idx])
    
    # else return []
    else: logPrior = np.array([])
    return(logPrior)

def _restrictfun(f, arg, free_idx, free_arg):
    """internal function not to be called from outside
    construction of file handles to restrict function"""
    # replace dummy arg 
    arg[free_idx] = free_arg
    # and evaluate
    val, dummy2 = f(arg) 
    return(val)

def _get_near_psd(A):
    """helper function to get closest definite matrix (if needed)"""
    if not _check_symmetric(A):
        C = (A + A.T)/2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0
        A = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    return(A)

def _correlation_from_covariance(covariance):
    """get correlation matrix from covariance matrix"""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return(correlation)

def _check_symmetric(a, tol=1e-8):
    """check if matrix is symatrical"""
    return(np.all(np.abs(a-a.T) < tol))
