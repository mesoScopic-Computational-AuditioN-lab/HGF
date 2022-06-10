""" Fuctions for model fitting and model simulation of the Hierarchical Gaussian Filter
code takes in stimulus states and simulates perception and prediction of an agent

Model implemented as discribed in: Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. Frontiers in human neuroscience, 8, 825.

Code adapted by Jorie van Haren (2021) """

# load nessecary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load extra (non exclusive) helper function
from HGF.hgf import _sgm

## Construct dataframe function

def constructDataframe(r, sim):
    """input fitted data r, and simulated data sim (from fitModel and simModel)
    returns a pandas dataframe for easy plotting"""
    # construct dataframe
    df_dict = {}
    df_dict['u'] = sim['u']
    df_dict['y'] = sim['y']
    df_dict['trial'] = np.arange(1, len(sim['u'])+1)

    # add fitted optimization parameters
    df_dict['fit_yhat']   = r['optim']['yhat']
    df_dict['fit_res']    = r['optim']['res']
    df_dict['fit_resAC']  = r['optim']['resAC']

    # loop and add all fitted trajectory parameters
    for item in r['traj'].keys():
        for lvl in range(len(r['traj'][item][0,:])):
            df_dict['fit_{}_lvl{}'.format(item, lvl+1)] = r['traj'][item][:,lvl]

    # loop and add all simulated trajectory parameters
    for item in sim['traj'].keys():
        for lvl in range(len(sim['traj'][item][0,:])):
            df_dict['sim_{}_lvl{}'.format(item, lvl+1)] = sim['traj'][item][:,lvl]

    # finally add this all to a dict
    df = pd.DataFrame(df_dict)
    
    # add priors
    df = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns).append(df, ignore_index=True)
    for lvl in range(len(r['traj']['mu'][0,:])):
        # fits
        df.loc[0, 'fit_mu_lvl{}'.format(lvl+1)] = r['p_prc']['mu_0'][lvl]
        df.loc[0, 'fit_sa_lvl{}'.format(lvl+1)] = r['p_prc']['sa_0'][lvl]
    for lvl in range(len(sim['traj']['mu'][0,:])):
        # simulations
        df.loc[0, 'sim_mu_lvl{}'.format(lvl+1)] = sim['p_prc']['mu_0'][lvl]
        df.loc[0, 'sim_sa_lvl{}'.format(lvl+1)] = sim['p_prc']['sa_0'][lvl]
    
    # sgm transform
    if 'binary' in r['c_prc']['model']: 
        df = df.assign(fit_mu_lvl2_sgm= _sgm(df['fit_mu_lvl2'], 1))
        df = df.assign(sim_mu_lvl2_sgm= _sgm(df['sim_mu_lvl2'], 1))
        
    # calculate higher and lower bound
    for lvl in range(len(r['traj']['mu'][0,:])):
        kwargs = {'fit_mu_lvl{}_upper'.format(lvl+1) : df['fit_mu_lvl{}'.format(lvl+1)] + np.sqrt(df['fit_sa_lvl{}'.format(lvl+1)])}
        df = df.assign(**kwargs)
        kwargs = {'fit_mu_lvl{}_lower'.format(lvl+1) : df['fit_mu_lvl{}'.format(lvl+1)] - np.sqrt(df['fit_sa_lvl{}'.format(lvl+1)])}
        df = df.assign(**kwargs)
        
    for lvl in range(len(sim['traj']['mu'][0,:])):
        kwargs = {'sim_mu_lvl{}_upper'.format(lvl+1) : df['sim_mu_lvl{}'.format(lvl+1)] + np.sqrt(df['sim_sa_lvl{}'.format(lvl+1)])}
        df = df.assign(**kwargs)
        kwargs = {'sim_mu_lvl{}_lower'.format(lvl+1) : df['sim_mu_lvl{}'.format(lvl+1)] - np.sqrt(df['sim_sa_lvl{}'.format(lvl+1)])}
        df = df.assign(**kwargs)
    return(df)


## Plotting functions

def plot_binary_expect(df, r, fit='sim'):
    """Function to plot binary expectations over all levels"""

    # configure plot size
    fig, ax = plt.subplots(r['c_prc']['n_levels'], 
                           1, 
                           sharex=True, 
                           figsize=(12, r['c_prc']['n_levels']*4))

    for lvl in np.arange(r['c_prc']['n_levels'], 0, -1):

        # sellect subplot
        pltnr = r['c_prc']['n_levels']-lvl

        if lvl > 1:
            # plot main results
            ax[pltnr].plot(df['{}_mu_lvl{}'.format(fit,lvl)])
            ax[pltnr].fill_between(df['trial'], 
                                   df['{}_mu_lvl{}_upper'.format(fit,lvl)], 
                                   df['{}_mu_lvl{}_lower'.format(fit,lvl)], 
                                   alpha=0.2)

            ax[pltnr].set_ylabel('μ{}'.format(lvl), fontsize=16)
            ax[pltnr].tick_params(axis='y', which='major', labelsize=16)

            ax[pltnr].set_title('Posterior expectations of x{}'.format(lvl), fontsize=18)
        else:

            # adjust to make better visible
            df.loc[df['y'] == 1, 'y'] = 0.96
            df.loc[df['y'] == 0, 'y'] = 0.04

            # plot actual and response
            ax[pltnr].scatter(df['trial'],df['u'], label='Stimuli', color='orange')
            ax[pltnr].scatter(df['trial'],df['y'], label='Response', color='green')

            # plot the mu expectation
            ax[pltnr].plot(df['{}_mu_lvl2_sgm'.format(fit)])

            # set legend and ylabel
            ax[pltnr].legend(fontsize=16)
            ax[pltnr].set_ylabel('u, y, s(μ2)', fontsize=16)
            ax[pltnr].tick_params(axis='y', which='major', labelsize=16)

            ax[pltnr].set_title('Inputs, Responses, and posterior expectations of input'.format(lvl), fontsize=18)

    # set xlabel for everything and title
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('Binary expectations\n',fontsize=22);
    plt.tight_layout()
    return(ax)
    
    
def plot_binary_learningrate(df, fit='sim'):
    """Function to plot learningrate for output level"""

    # configure plot size
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True, 
                           figsize=(12, 8), 
                           gridspec_kw={'height_ratios': [3, 1]})

    # adjust to make better visible
    df.loc[df['y'] == 1, 'y'] = 0.96
    df.loc[df['y'] == 0, 'y'] = 0.04

    # plot actual and response
    ax[0].scatter(df['trial'],df['u'], label='Stimuli', alpha=0.5, color='orange')
    ax[0].scatter(df['trial'],df['y'], label='Response', alpha=0.5, color='green')

    # plot the mu expectation
    ax[0].plot(df['{}_mu_lvl2_sgm'.format(fit)], color='black', lw=2, ls='--',  alpha=0.5)        
    ax[1].plot(df['sim_wt_lvl1'], color='red', lw=2)

    # set legend and ylabel
    ax[0].legend(fontsize=16)
    ax[0].set_ylabel('u, y, s(μ2)', fontsize=16)
    ax[0].tick_params(axis='y', which='major', labelsize=16)
    ax[0].set_title('Inputs, Responses, and posterior expectations of input', fontsize=18)

    ax[1].set_ylabel('lr', fontsize=16)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].set_title('Learning Rate', fontsize=18)
    
    # set xlabel for everything and title
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('Learning rate',fontsize=22);
    
    plt.tight_layout()
    return(ax)
    
    
def plot_expect(df, r, fit='sim', pres_post=True):
    """Function to plot expectations over all levels"""

    # configure plot size
    fig, ax = plt.subplots(r['c_prc']['n_levels'], 
                           1, 
                           sharex=True, 
                           figsize=(12, r['c_prc']['n_levels']*4))

    for lvl in np.arange(r['c_prc']['n_levels'], 0, -1):

        # sellect subplot
        pltnr = r['c_prc']['n_levels']-lvl
        
        # plot main results
        if pres_post == True or lvl > 1:
            ax[pltnr].plot(df['{}_mu_lvl{}'.format(fit,lvl)], alpha=0.8)
            ax[pltnr].fill_between(df['trial'], 
                                   df['{}_mu_lvl{}_upper'.format(fit,lvl)], 
                                   df['{}_mu_lvl{}_lower'.format(fit,lvl)], 
                                   alpha=0.3)

        if lvl > 1:
            # set labels 
            ax[pltnr].set_ylabel('μ{}'.format(lvl), fontsize=16)
            ax[pltnr].tick_params(axis='y', which='major', labelsize=16)
            ax[pltnr].set_title('Posterior expectations of x{}'.format(lvl), fontsize=18)
        
        else:
            # plot actual and response
            ax[pltnr].scatter(df['trial'],df['u'], label='Stimuli', alpha=0.8, s=3, color='orange')
            ax[pltnr].scatter(df['trial'],df['y'], label='Response', alpha=0.8, s=3, color='green')

            # set legend and ylabel
            ax[pltnr].legend(fontsize=16)
            ax[pltnr].set_ylabel('u, y, s(μ2)', fontsize=16)
            ax[pltnr].tick_params(axis='y', which='major', labelsize=16)

        ax[pltnr].set_title('Inputs, Responses, and posterior expectations of input'.format(lvl), fontsize=18)

    # set xlabel for everything and title
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('Expectations\n',fontsize=22);
    plt.tight_layout()
    return(ax)
    
    
def plot_learningrate(df, fit='sim', alpha_mu=0.5):
    """Function to plot learningrate for output level"""

    # configure plot size
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True, 
                           figsize=(12, 8), 
                           gridspec_kw={'height_ratios': [3, 1]})

    # plot actual and response
    ax[0].scatter(df['trial'],df['u'], label='Stimuli', alpha=0.5, s=4, color='orange')
    ax[0].scatter(df['trial'],df['y'], label='Response', alpha=0.5, s=4, color='green')

    # plot the mu expectation
    ax[0].plot(df['{}_mu_lvl1'.format(fit)], color='black', lw=1, ls='--',  alpha=alpha_mu)
    ax[1].plot(df['sim_wt_lvl1'], color='red', lw=2)

    # set legend and ylabel
    ax[0].legend(fontsize=16)
    ax[0].set_ylabel('u, y, s(μ2)', fontsize=16)
    ax[0].tick_params(axis='y', which='major', labelsize=16)
    ax[0].set_title('Inputs, Responses, and posterior expectations of input', fontsize=18)
    
    ax[1].set_ylabel('lr', fontsize=16)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].set_title('Learning Rate', fontsize=18)
    
    # set xlabel for everything and title
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('Learning rate',fontsize=22);
    plt.tight_layout()
    return(ax)


def plot_precision_weights(df, fit='sim'):
    """Plot the precision weights over all levels.
    Note that alient events is reflected in the precision weights
    input: df, optional fit ('sim' or 'fit')
    returns: plt plot"""

    # set image settings
    fig, ax = plt.subplots(figsize=(12, 6))

    # set columns of interest and number of levels
    colofintr = [s for s in df.columns if '{}_wt'.format(fit) in s]
    n_levels = ['1st level', '2nd level', '3rd level', '4th level', '5th level',
                '6th level', '7th level', '8th level', '9th level', '10th level'][:len(colofintr)]

    # plot precision weights 
    ax.plot(df[colofintr], lw=2.5)

    # set x and y label para
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('Weights', fontsize=16)
    plt.yticks(fontsize=16)

    # pimp plot
    ax.legend(n_levels, fontsize=14)
    plt.suptitle('Precision weights', fontsize=22)
    plt.tight_layout()
    return(ax)


def plot_residualdiag(r):
    """Plot residuals / difference between pred and response.
    Usefull to check for patterns (indicating model failed to capture ellements of the data)
    input dictonairy and returns plt plot"""
    # configure plot size
    fig, ax = plt.subplots(3, 
                           1, 
                           sharex=False, 
                           figsize=(12, 12))


    # plot residual time series and pimp
    ax[0].plot(r['optim']['res'])
    ax[0].tick_params(labelsize=16)
    ax[0].set_title('Residuals time series', fontsize=18, fontweight='bold')
    ax[0].set_ylabel('Residuals', fontsize=16)
    ax[0].set_xlabel('Trial nr.', fontsize=16)

    # plot risidual autocorrelation 
    ax[1].acorr(r['optim']['res'], maxlags = int(len(r['optim']['res'])/2), lw=2.5)
    ax[1].tick_params(labelsize=16)
    ax[1].set_title('Residuals autocorrelation', fontsize=18, fontweight='bold')
    ax[1].set_ylabel('Coeff.', fontsize=16)
    ax[1].set_xlabel('Lag', fontsize=16)

    # plot risiduals vs predictions
    ax[2].scatter(r['optim']['yhat'], r['optim']['res'])
    ax[2].tick_params(labelsize=16)
    ax[2].set_title('Scatter residuals vs predictions', fontsize=18, fontweight='bold')
    ax[2].set_ylabel('Residuals', fontsize=16)
    ax[2].set_xlabel('Predictions', fontsize=16)

    # set title and layout
    plt.suptitle('Residuals diagnostics\n',fontsize=22);
    plt.tight_layout()
    return(ax)

## Print function

def printfitmodel(r):
    """print modelfit results"""
    
    print('\n\nRESULTS:')
    modeltype = ['perceptual model', 'observation model']
    modelkeys = ['p_prc', 'p_obs']
    if r['p_obs']['ptrans'].any(): n_models = len(modeltype)
    else:                          n_models = len(modeltype)-1

    for modt in range(n_models):
        print('\nParameter estimates - {}:'.format(modeltype[modt]))
        for item in r[modelkeys[modt]].keys():
            if item not in ['p', 'ptrans']: print(' {}: \t {}'.format(item, r[modelkeys[modt]][item]))

    print('\nMODEL QUALITY:')
    for i in [['LME', 'more'], ['AIC', 'less'], ['BIC', 'less']] :
        print(' {}: \t {} \t\t ({} is better)'.format(i[0], r['optim'][i[0]], i[1]))

    return
