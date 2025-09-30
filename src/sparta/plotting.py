"Module for plotting outputs of SPaRTA."

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
from scipy.stats import norm as norm_dist
from scipy.stats import beta as beta_dist
from . misc import compute_Lya_cross_section

def plot_cross_section(
    T=0.,
    nu_min=None,
    nu_max=1.e-3,
    velocity=0.,
    flag='LORENTZIAN',
    ax=None,
    **kwargs
    ):

    """
    Plot Lyman alpha cross section as a function of frequency.
    
    Parameters
    ----------
    T: float, optional
        Temperature of the IGM (in Kelvin degrees)
    nu_min: float, optional
        Minimum frequency for which the cross-section will be displayed,
        with respect to the center of the line, in units of Lyman alpha 
        frequency.
    nu_max: float, optional
        Maximum frequency for which the cross-section will be displayed,
        with respect to the center of the line, in units of Lyman alpha 
        frequency.
    velocity: float, optional
        The relative velocity of the interacting hydrogen atom, in units of c.
        The relative velocity causes a Doppler shift in the profile.
    flag: string, optional
        Type of cross-section to be used. Options are (default is 'Lorenzian'):
            - 'LORENTZIAN': A simple Lorentzian profile is used when T=0. 
            For a finite temperature, the Lorentzian profile is convolved 
            with a Gaussian to yield to Voigt profile.
            - 'PEEBLES': Cross-section from Peebles, P.J.E. (1993) Principles 
            of Physical Cosmology (see also Eq. 49 in arXiv: 1704.03416).
            Only works for T=0.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    if nu_min is None:
        nu_min = -nu_max
        nu_array_zero = 1. + np.linspace(nu_min,nu_max,1000)
    else:
        nu_array_zero = 1. + np.logspace(np.log10(nu_min),np.log10(nu_max),1000)
    nu_array = nu_array_zero/(1.-velocity)
    sigma_array = np.array([compute_Lya_cross_section(nu,T,flag=flag) for nu in nu_array]) # m^2
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Multiply by 1e4 to show cross-section in cm^2
    ax.semilogy(nu_array_zero-1.,1e4*sigma_array,**kwargs)
    if nu_min > 0:
        ax.set_xscale('log')
    ax.set_xlim([nu_min,nu_max])
    ax.set_xlabel('$\\nu/\\nu_\\alpha-1$',fontsize=25)
    ax.set_ylabel('$\\sigma_\\alpha\\,[\\mathrm{cm^2}]$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_scatter_all_photons(
    sim_data,
    x_axis='distance',
    ax = None,
    **kwargs
    ):
        
    """
    Plot the scatter of simulation data points as a function
    of the redshift of the emitter.
    
    Parameters
    ----------
    x_axis: string, optional
        Determines the type of x-axis to be displayed.
        There are two options:
            - 'distance': comoving straight-line distance between z_abs 
                            and z_em, normalized by the diffusion scale at
                            z_abs.
            - 'redshift': the redshift of the emitter, z_em.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    # Check input
    if not (x_axis == 'distance' or x_axis == 'redshift'):
        raise ValueError("'x_axis' can only be 'distance' or 'redshift'.")
    if x_axis == 'distance':
        x_list = sim_data.x_list
    else:
        x_list = np.array([sim_data.cosmo_params.R_SL_inverse(sim_data.z_abs,x_em*sim_data.cosmo_params.r_star(sim_data.z_abs)) for x_em in sim_data.x_list])
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot scatter of the data
    ax.scatter(x_list,sim_data.y_list,**kwargs)
    # Prettify plot
    if sim_data.quantity == 'distance':
        ax.set_ylabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif sim_data.quantity == 'velocity':
        ax.set_ylabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if x_axis == 'distance':
        ax.set_xlabel('$R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})/r_\\star(z_\\mathrm{abs})$',fontsize=25)
    else:
        ax.set_xlabel('$z_\\mathrm{em}$',fontsize=25)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_fit(
    sim_data,
    x_em,
    x_res = 0.1,
    ax = None,
    **kwargs
    ):
        
    """
    Plot the analytical fit to the 1D histogram at z_em.
    
    Parameters
    ----------
    x_em: float
        Comoving straight-line distance between z_abs and z_em, normalized 
        by the diffusion scale at z_abs.
    x_res: float, optional
        Resolution (width) of the x_em bins.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    
    """
    
    dist_params, data_lims = sim_data.find_fitting_params(x_em=x_em,x_res=x_res)
    if sim_data.quantity == 'distance':
        (alpha, beta) = dist_params
        x_array = np.linspace(0.,1.,1000)
    elif sim_data.quantity == 'velocity':
        (mu, sigma) = dist_params
        x_array = np.linspace(data_lims[0],data_lims[1],1000)
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot the analytical fit
    if sim_data.quantity == 'distance':
        ax.plot(x_array, beta_dist.pdf(x_array,alpha,beta),**kwargs)
        # Prettify plot
        ax.set_ylabel('Radial distribution',fontsize=25)
        ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif sim_data.quantity == 'velocity':
        ax.plot(x_array, norm_dist.pdf(x_array,mu,sigma), **kwargs)
        ax.set_ylabel('Velocity distribution',fontsize=25)
        ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_histogram(
    histogram_data,
    z_em=None,
    x_em=None,
    ax = None,
    **kwargs
    ):
        
    """
    Plot the 1D histogram of the data at a given emitter redshift.
    The plotted histogram is normalized to 1.
    
    Parameters
    ----------
    z_em: float, optional
        The redshift of the emitter. Cannot be passed if x_em is 
        also passed.
    x_em: float, optional
        Comoving straight-line distance between z_abs and z_em, normalized 
        by the diffusion scale at z_abs. Cannot be passed if z_em is 
        also passed.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.bar`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    # Check input
    if x_em == None and z_em == None:
        raise ValueError("Either 'x_em' or 'z_em' must be specified.")
    if (not x_em == None) and (not z_em == None):
        raise ValueError("'x_em' and 'z_em' cannot be both specified. Choose one of them.")
    # Find 1D data that corresponds to input
    if not z_em == None:
        x_em = histogram_data.cosmo_params.R_SL(histogram_data.z_abs,z_em)/histogram_data.cosmo_params.r_star(histogram_data.z_abs)
    if x_em > np.max(histogram_data.x_bins):
        raise ValueError(f"Requested input corresponds to x_em={x_em}, but maximum x_em in histogram data is {np.max(histogram_data.x_bins)}")
    elif x_em < np.min(histogram_data.x_bins):
        raise ValueError(f"Requested input corresponds to x_em={x_em}, but minimum x_em in histogram data is {np.min(histogram_data.x_bins)}")
    else:
        x_em_arg = np.argmin(np.abs(histogram_data.x_bins-x_em))
    # Extract 1D distribution
    f_array = histogram_data.H_matrix[x_em_arg,:] # dimensionless
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    if not "align" in kwargs.keys():
        kwargs["align"] = "edge"
    if not "edgecolor" in kwargs.keys():
        kwargs["edgecolor"] = "black"
    # Find edges
    dr = np.diff(histogram_data.y_bins)[0]
    edges = np.concatenate((np.array([histogram_data.y_bins[0]-dr/2.]),
                            (histogram_data.y_bins[1:]+histogram_data.y_bins[:-1])/2.,
                            np.array([histogram_data.y_bins[-1]+dr/2.])))
    # Plot 1D histogram of the data, normalized to 1
    if histogram_data.quantity == 'distance':
        r_array = histogram_data.y_bins # dimensionless
        # Ensure that the bin at 0 and 1 are empty
        f_array[1] += f_array[0]
        f_array[0] = 0.
        if not histogram_data.sim_params.STRAIGHT_LINE:
            f_array[-2] += f_array[-1]
            f_array[-1] = 0.
        # Normalize
        f_array /= intg.simpson(f_array, x=r_array) # dimensionless
        # Plot!
        ax.bar(edges[:-1],f_array,width=np.diff(edges),**kwargs)
        # Prettify plot
        ax.set_ylabel('Radial distribution',fontsize=25)
        ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif histogram_data.quantity == 'velocity':
        v_array = histogram_data.y_bins # dimensionless
        # Normalize
        f_array /= intg.simpson(f_array, x=v_array) # dimensionless
        # Plot!
        ax.bar(edges[:-1],f_array,width=np.diff(edges),**kwargs)
        ax.set_ylabel('Velocity distribution',fontsize=25)
        ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_histogram_fit(
    histogram_data,
    z_em=None,
    x_em=None,
    ax = None,
    **kwargs
    ):
        
    """
    Plot the analytical fit to the 1D histogram at z_em.
    
    Parameters
    ----------
    z_em: float, optional
        The redshift of the emitter. Cannot be passed if x_em is 
        also passed.
    x_em: float, optional
        Comoving straight-line distance between z_abs and z_em, normalized 
        by the diffusion scale at z_abs. Cannot be passed if z_em is 
        also passed.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """

    # Check input
    if x_em == None and z_em == None:
        raise ValueError("Either 'x_em' or 'z_em' must be specified.")
    if (not x_em == None) and (not z_em == None):
        raise ValueError("'x_em' and 'z_em' cannot be both specified. Choose one of them.")
    # Find 1D data that corresponds to input z_em
    if not z_em == None:
        if histogram_data.quantity == 'distance':
            f_array, x_array, (alpha, beta) = histogram_data.fit_histogram(z_em=z_em)
        else:
            f_array, x_array, (mu, sigma) = histogram_data.fit_histogram(z_em=z_em)
    else:
        if histogram_data.quantity == 'distance':
            f_array, x_array, (alpha, beta) = histogram_data.fit_histogram(x_em=x_em)
        else:
            f_array, x_array, (mu, sigma) = histogram_data.fit_histogram(x_em=x_em)
    # Get more samples to draw nice functions
    x_array_new = np.linspace(x_array[0],x_array[-1],1000)
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot the analytical fit
    if histogram_data.quantity == 'distance':
        ax.plot(x_array_new, beta_dist.pdf(x_array_new,alpha,beta),**kwargs)
        # Prettify plot
        ax.set_ylabel('Radial distribution',fontsize=25)
        ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif histogram_data.quantity == 'velocity':
        ax.plot(x_array_new, norm_dist.pdf(x_array_new,mu,sigma), **kwargs)
        ax.set_ylabel('Velocity distribution',fontsize=25)
        ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax