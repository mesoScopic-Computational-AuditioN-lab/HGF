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
from hgf_config import *
from hgf import *
from hgf_pres import *

# load extra (non exclusive) helper function
from hgf import _unpack_para

#######################
## MAIN FIT FUNCTION ##
#######################

def simModel(inputs, prc_model, prc_pvec,
             obs_model=False,
             obs_pvec=False,
             overwrite_opt=False,
             seed=np.random.randint(99999)):
    """
    Function to simulate responses and/or perceptual states.
    given perceptual and observational models and input into the system
    
    input:  inputs    = array of inputs
            prc_model = perceptual model (e.g. hgf_binary, hgf, ehgf)
            prc_pvec  = array of perceptual model parameter values
            obs_model = (optional) observational model or non (unitsq_sgm, gaussian_obs)
                        default false: no response will be simulated
            obs_pvec  = (optional) array of observation model parameter values
            overwrite_opt = default False, or a dictionary with personal settings instead of 
                            retrieving them from _config
                         - Dict should have dict['c_prc'], dict['c_obs'], and/or dict['c_opt']
                         - In here you may place keys with own options
                         - e.g. overwrite_optr['c_prc']['rhomu'] = np.array(['np.nan, 0.5, 0.5'])
            seed      = random number seed
            
    returns: dict r with perceptual states and responses"""
    
    # set config linkings
    configz = {hgf_binary:hgf_binary_config,
               ehgf_binary:ehgf_binary_config,
               unitsq_sgm:unitsq_sgm_config,
               hgf:hgf_config,
               gaussian_obs:gaussian_obs_config}
    simz    = {unitsq_sgm:unitsq_sgm_sim,
               gaussian_obs:gaussian_obs_sim}

    # create empty dict to store everything
    r = {}
    r['u'] = np.array(inputs)   # store inputs

    # check for ignored trials and irregular trials
    r['ign'] = np.argwhere(np.isnan(r['u']))
    print('Ignored trials: {}'.format(r['ign']))

    # set perceptual model
    r['c_sim']              = {}
    r['c_sim']['prc_model'] = prc_model

    # run config function to set config settings
    r['c_prc']              = configz[prc_model]()   

    # override with our own settings
    if overwrite_opt != False:
        for item in ['c_prc', 'c_sim']:
            if item not in overwrite_opt: overwrite_opt[item] = {}
            r[item] = {**r[item], **overwrite_opt[item]}
    
    #  check if levels and length prc_pvec are consistent -- NEW FUNCTION ADD PLEASE
    if round(len(prc_pvec)/5) != r['c_prc']['n_levels']:
        r = _adjust_lvls(prc_pvec, r)
    
    # unpack pvec variables
    r['p_prc']              = _unpack_para(prc_pvec, r)
    r['p_prc']['p']         = prc_pvec 

    # compute perceptual states
    r['traj'], infStates    = prc_model(r, r['p_prc']['p'])

    # if obs model and pvec is not false we simulate responses
    if (obs_model != False) and (obs_pvec != False):

        # store obs parameters
        r['c_sim']['obs_model'] = obs_model
        r['p_obs']              = {}
        r['p_obs']['p']         = obs_pvec           # these two are not standardized yet and are fully based on unitsq_sgm
        r['p_obs']['ze']        = obs_pvec           # these two are not standardized yet and are fully based on unitsq_sgm
        r['c_obs']              = configz[obs_model]
        r['c_sim']['seed']      = seed

        # override obs with own
        if overwrite_opt != False:
            if 'c_obs' in overwrite_opt: r['c_obs'] = {**r['c_obs'], **overwrite_opt['c_obs']}
        
        # simulate decisions
        r['y']                  = simz[obs_model](r, infStates, r['p_obs']['p'])
    return(r)


def unitsq_sgm_sim(r, infStates, p, predpos=0):
    """simple function to simulate observations from distribution
    optional input predpos can be set to 0 to instead use posteriors instead of predictions"""
    # decision temp
    ze = p
    
    # trajectory beliefs at lvl 1
    states = infStates[:,0,predpos]
    
    # apply unit-square sigmoid to inferred state
    prob = np.divide(states**ze , states**ze + (1-states)**ze) 
    
    # set random seed
    np.random.seed(r['c_sim']['seed'])
    
    # and simulate
    y = np.random.binomial(1, prob)
    return(y)



def gaussian_obs_sim(r, infStates, p, predpos=0):
    """simple function to simulate observations from distribution
    optional input predpos can be set to 0 to instead use posteriors instead of predictions"""
    # decision temp
    ze = p
    
    # trajectory beliefs at lvl 1
    muhat = infStates[:,0,predpos]
    
    # number of trials
    n = len(muhat)
    
    # set random seed
    np.random.seed(r['c_sim']['seed'])
    
    # and simulate
    y = muhat + np.sqrt(ze) * np.random.randn(n)
    return(y)


def _adjust_lvls(prc_pvec, r):
    """internal helper function, input the pvec array and r dict
    asks if you want to adjust levels and returns"""
    
    # print message
    print("\nNumber of levels (depth) inconsistent with length indicated by 'prc_pvec'\n n_levels: {}\n prc_pvec depth: {}".format(r['c_prc']['n_levels'], 
                                                                                                      round(len(prc_pvec)/5)))
    print("\n...\n\nSetting new number of levels to {}\n(Make sure no error was made in setting up prc_pvec)".format(round(len(prc_pvec)/5)))
    
    # do the actual adjustment
    r_levels = {'n_levels':round(len(prc_pvec)/5)}
    r['c_prc'] = {**r['c_prc'], **r_levels}
    
    return(r)