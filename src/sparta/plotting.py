"Module for plotting outputs of SPaRTA."

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
from numpy.linalg import norm
from scipy.stats import norm as norm_dist
from scipy.stats import beta as beta_dist
from . import correlations
from . Lyman_alpha import compute_Lya_cross_section
from . post_processing import multiple_scattering_window_function
from . inputs import COSMO_PARAMS

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

def plot_beta_distribution(
    alpha,
    beta,
    ax = None,
    **kwargs
    ):
        
    """
    Plot the analytical fit to the 1D histogram at x_em.
    
    Parameters
    ----------
    alpha: float
        First parameter of the window function.
    beta: float
        Second parameter of the window function.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    
    """
    
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot the analytical fit
    x_array = np.linspace(0.,1.,1000)
    ax.plot(x_array, beta_dist.pdf(x_array,alpha,beta),**kwargs)
    # Prettify plot
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

#%% CLASS_OUTPUT plots

def plot_RMS(
    CLASS_OUTPUT,
    r_smooth,
    kind = "density_m",
    z_min = 0.,
    z_max = 35.,
    ax = None,
    **kwargs
):
    """
    Plot the RMS of a smoothed field as a function of redshift.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    r_smooth: float
        The smoothing radius in Mpc.
    kind: str, optional
        The kind of field for which the RMS is computed: options are "density_m", "density_b"
        "v_parallel" and "v_perp" (the RMS of the last two is the same,
        hence "velocity" is also accepted). Default is "density_m".
    z_min: float, optional
        The minimum redshift in the plot. Default is 0.
    z_max: float, optional
        The maximum redshift in the plot. Default is 35.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """

    z_array = np.linspace(z_min,z_max,100)
    rms_array = np.zeros_like(z_array)
    for ind, z in enumerate(z_array):
        rms_array[ind] = correlations.compute_RMS(
            CLASS_OUTPUT = CLASS_OUTPUT,
            z = z,
            r_smooth = r_smooth,
            kind = kind
        )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    ax.plot(z_array,rms_array,**kwargs)
    ax.set_xlabel('$z$',fontsize=25)
    ax.set_ylabel('RMS',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([z_array.min(),z_array.max()])
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_Pearson_coefficient(
    CLASS_OUTPUT,
    z,
    r_smooth,
    kinds = ("density_m","density_m"),
    r_min = 0.,
    r_max = 10.,
    evolve_z2 = False,
    log_x = False,
    ax = None,
    **kwargs
):
    """
    Plot the Pearson coefficient for two smoothed fields as a function of comoving distance.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    z: float
        The redshift of the field of kind1 (see below). If evolve_z2 is True,
        it is also the redshift of the field of kind2.
    r_smooth: float
        The smoothing radius in Mpc.
    kinds: tuple, optional
        A tuple of the form (kind1,kind2).
        Each kind is a string specifying the kind of the field for which the 
        Pearson coefficient is evaluated: options are "density_m", "density_b"
        "v_parallel" and "v_perp". Default is ("density_m","density_m").
    r_min: float, optional
        The minimum comoving distance in the plot, in Mpc. Default is 0.
    r_max: float, optional
        The maximum comoving distance in the plot, in Mpc. Default is 10.
    evolve_z2: bool, optional
        If this flag is true, z2 (the redshift of the field of kind2) is evolved with the comoving 
        distance. Othwerwise, it is set to be z.
    log_x: bool, optional
        Whether to plot the distance axis in the plot with logarithmic scale. Default is False.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    if evolve_z2:
        cosmo_params = COSMO_PARAMS(
            h =  CLASS_OUTPUT.h(),
            Omega_m = CLASS_OUTPUT.Omega0_cdm() + CLASS_OUTPUT.Omega_b(),
            Omega_b = CLASS_OUTPUT.Omega_b(),
        )
    kind1, kind2 = kinds
    if log_x:
        r_array = np.logspace(np.log10(r_min),np.log10(r_max),100)
    else:
        r_array = np.linspace(r_min,r_max,100)
    rho_array = np.zeros_like(r_array)
    for ind, r in enumerate(r_array):
        if evolve_z2:
            z2 = cosmo_params.R_SL_inverse(z,r)
        else:
            z2 = z
        rho_array[ind] = correlations.compute_Pearson_coefficient(
            CLASS_OUTPUT = CLASS_OUTPUT,
            z1 = z,
            z2 = z2,
            r = r,
            r_smooth = r_smooth,
            kinds_list = [kinds]
        )[f"{kind1},{kind2}"]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    ax.plot(r_array,rho_array,**kwargs)
    ax.set_xlabel('$r\\,[\\mathrm{Mpc}]$',fontsize=25)
    ax.set_ylabel('Pearson correlation coefficient',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([r_array.min(),r_array.max()])
    if log_x:
        ax.set_xscale("log")
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_correlation_function(
    CLASS_OUTPUT,
    z,
    r_smooth,
    kinds = ("density_m","density_m"),
    r_min = 0.,
    r_max = 10.,
    evolve_z2 = False,
    log_x = False,
    ax = None,
    **kwargs
):
    """
    Plot the correlation function for two smoothed fields as a function of comoving distance.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    z: float
        The redshift of the field of kind1 (see below). If evolve_z2 is True,
        it is also the redshift of the field of kind2.
    r_smooth: float
        The smoothing radius in Mpc.
    kinds: tuple, optional
        A tuple of the form (kind1,kind2).
        Each kind is a string specifying the kind of the field for which the 
        Pearson coefficient is evaluated: options are "density_m", "density_b"
        "v_parallel" and "v_perp". Default is ("density_m","density_m").
    r_min: float, optional
        The minimum comoving distance in the plot, in Mpc. Default is 0.
    r_max: float, optional
        The maximum comoving distance in the plot, in Mpc. Default is 10.
    evolve_z2: bool, optional
        If this flag is true, z2 (the redshift of the field of kind2) is evolved with the comoving 
        distance. Othwerwise, it is set to be z.
    log_x: bool, optional
        Whether to plot the distance axis in the plot with logarithmic scale. Default is False.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """

    if evolve_z2:
        cosmo_params = COSMO_PARAMS(
            h =  CLASS_OUTPUT.h(),
            Omega_m = CLASS_OUTPUT.Omega0_cdm() + CLASS_OUTPUT.Omega_b(),
            Omega_b = CLASS_OUTPUT.Omega_b(),
        )
    kind1, kind2 = kinds
    if log_x:
        r_array = np.logspace(np.log10(r_min),np.log10(r_max),100)
    else:
        r_array = np.linspace(r_min,r_max,100)
    xi_array = np.zeros_like(r_array)
    for ind, r in enumerate(r_array):
        if evolve_z2:
            z2 = cosmo_params.R_SL_inverse(z,r)
        else:
            z2 = z
        xi_array[ind] = correlations.compute_correlation_function(
            CLASS_OUTPUT = CLASS_OUTPUT,
            z1 = z,
            z2 = z2,
            r = r,
            r_smooth = r_smooth,
            kind1 = kind1,
            kind2 = kind2,
        )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    ax.plot(r_array,xi_array,**kwargs)
    ax.set_xlabel('$r\\,[\\mathrm{Mpc}]$',fontsize=25)
    ax.set_ylabel('Correlation function',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([r_array.min(),r_array.max()])
    if log_x:
        ax.set_xscale("log")
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax
    

#%% PHOTON_POINTS_DATA plots

def plot_photon_trajectory(
    photon_points_data,
    scale=5.,
    ax = None,
    **kwargs
    ):
    
    """
    Plot the trajectory of this photon.
    
    Parameters
    ----------
    photon_points_data: PHOTON_POINTS_DATA
        The object that contains the data points of the photon.
    scale: float, optional
        The scale for this plot in Mpc.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    x_list = []
    y_list = []
    z_list = []
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='3d'))
    else:
        fig = ax.figure
    for i in range(len(photon_points_data.points_data)):
        z_i_data = photon_points_data.points_data[i]
        x_list.append(z_i_data.position_vector[0])
        y_list.append(z_i_data.position_vector[1])
        z_list.append(z_i_data.position_vector[2])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    z_array = np.array(z_list)
    ax.plot(x_array,y_array,z_array,**kwargs)
    ax.set_xlabel('$X\\,[\\mathrm{Mpc}]$',fontsize=15)
    ax.set_ylabel('$Y\\,[\\mathrm{Mpc}]$',fontsize=15)
    ax.set_zlabel('$Z\\,[\\mathrm{Mpc}]$',fontsize=15)
    ax.set_xlim([-scale,scale])
    ax.set_ylim([-scale,scale])
    ax.set_zlim([-scale,scale])
    ax.set_xticks(np.arange(-scale,scale+scale/5.,scale/5.))
    ax.set_yticks(np.arange(-scale,scale+scale/5.,scale/5.))
    ax.set_zticks(np.arange(-scale,scale+scale/5.,scale/5.))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    if "label" in kwargs:
        ax.legend(fontsize=15)
    # Return output
    return fig, ax

def plot_apparent_frequency(
    photon_points_data,
    ax = None,
    **kwargs
    ):

    """
    Plot the evolution of the apparent frequncy of this photon 
    in the gas frame.
    
    Parameters
    ----------
    photon_points_data: PHOTON_POINTS_DATA
        The object that contains the data points of the photon.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    z_list = []
    nu_list = []
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    for i in range(len(photon_points_data.points_data)):
        z_i_data = photon_points_data.points_data[i]
        z_list.append(z_i_data.redshift)
        nu_list.append(z_i_data.apparent_frequency)
    z_array = np.array(z_list)
    nu_array = np.array(nu_list)
    ax.loglog((z_array-photon_points_data.z_abs)/(1.+photon_points_data.z_abs),nu_array-1.,**kwargs)
    ax.set_xlabel('$(z-z_\\mathrm{abs})/(1+z_\\mathrm{abs})$',fontsize=25)
    ax.set_ylabel('$\\nu/\\nu_\\alpha-1$',fontsize=25)
    ax.set_xlim(photon_points_data.sim_params.Delta_z_initial,photon_points_data.sim_params.nu_stop-1.)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

def plot_distance(
    photon_points_data,
    intermediate_pts = True,
    ax = None,
    **kwargs
    ):
    
    """
    Plot the distance evolution of this photon with respect to the 
    absorption point.
    
    Parameters
    ----------
    photon_points_data: PHOTON_POINTS_DATA
        The object that contains the data points of the photon.
    intermediate_pts: bool, optional
        If this flag is True, then intermediate points (i.e. at z') are
        also shown. If false, then only the points in which the photon
        has scattered, z_i, are shown.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """
    
    z_list = []
    r_list = []
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    for i in range(len(photon_points_data.points_data)):
        z_i_data = photon_points_data.points_data[i]
        z_list.append(z_i_data.redshift)
        r_list.append(norm(z_i_data.position_vector)) # Mpc
        if intermediate_pts:
            if i < len(photon_points_data.points_data)-1:
                z_i_plus1_data = photon_points_data.points_data[i+1]
                z_i_plus1 = z_i_plus1_data.redshift
                z_i = z_i_data.redshift
                # N is the number of grid points between z_i and z_{i+1}
                N = int(photon_points_data.cosmo_params.R_SL(z_i,z_i_plus1)/photon_points_data.sim_params.Delta_L)
                for n in np.arange(1,N):
                    z_prime = photon_points_data.cosmo_params.R_SL_inverse(z_i,n*photon_points_data.sim_params.Delta_L)
                    z_list.append(z_prime)
                    w = n/N
                    z_prime_position_vector = (1.-w)*z_i_data.position_vector # Mpc
                    z_prime_position_vector += w*z_i_plus1_data.position_vector # Mpc
                    r_list.append(norm(z_prime_position_vector)) # Mpc
    z_array = np.array(z_list)
    r_array = np.array(r_list)
    ax.loglog((z_array-photon_points_data.z_abs)/(1.+photon_points_data.z_abs),r_array,**kwargs)
    ax.set_xlabel('$(z-z_\\mathrm{abs})/(1+z_\\mathrm{abs})$',fontsize=25)
    ax.set_ylabel('Distance [Mpc]',fontsize=25)
    ax.set_xlim(photon_points_data.sim_params.Delta_z_initial,photon_points_data.sim_params.nu_stop-1.)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax

#%% SIM_DATA plots

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
    sim_data: SIM_DATA
        The object that contains all the data from the simulation.
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
        ax.set_ylabel('$y\\equiv r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif sim_data.quantity == 'velocity':
        ax.set_ylabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if x_axis == 'distance':
        ax.set_xlabel('$x_\\mathrm{em}\\equiv R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})/R_*(z_\\mathrm{abs})$',fontsize=25)
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
    Plot the analytical fit to the 1D histogram at x_em.
    
    Parameters
    ----------
    sim_data: SIM_DATA
        The object that contains all the data from the simulation.
    x_em: float
        Comoving straight-line distance between z_abs and z_em, normalized 
        by the diffusion scale at z_abs.
    x_res: float, optional
        Resolution (width) of the x_em bins. Default is 0.1.
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
        ax.set_xlabel('$y\\equiv r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
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

def plot_window_function(
    alpha = None,
    beta = None,
    sim_data = None,
    x_em = None,
    x_res = None,
    ax = None,
    **kwargs
    ):
    """
    Plot the multiple scattering window function as a function of kR. It is

    M(alpha,beta,kR) = 2F3( (2+alpha)/2, (3+alpha)/2 ; 5/2, (2+alpha+beta)/2 ,(3+alpha+beta)/2 ; -1/4 * kR^2)

    If alpha and beta are not provided, then sim_data and x_em must be provided, and alpha and beta will be inferred
    from the simulation data.
    
    Parameters
    ----------
    alpha: float, optional
        First parameter of the window function.
    beta: float, optional
        Second parameter of the window function.
    sim_data: SIM_DATA, optional
        The object that contains all the data from the simulation. Must be of type 'distance'.
    x_em: float, optional
        Comoving straight-line distance between z_abs and z_em, normalized 
        by the diffusion scale at z_abs.
    x_res: float, optional
        Resolution (width) of the x_em bins. Default is 0.1.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.
    
    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    
    """

    if (alpha is None) and (beta is None):
        if not sim_data.quantity == "distance":
            raise TypeError("sim_data is not of type 'distance'! The window function can be plotted only for data of type 'distance'.")
        dist_params, data_lims = sim_data.find_fitting_params(x_em=x_em,x_res=x_res)
        (alpha, beta) = dist_params
    elif not x_em is None:
        raise Exception("x_em should not be specified if alpha and beta are specified.")
    kR_array = np.linspace(0,30,200)
    M_array = multiple_scattering_window_function(kR_array,alpha,beta)
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot the window function
    ax.plot(kR_array,M_array,**kwargs)
    # Prettify plot
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([np.min(kR_array),np.max(kR_array)])
    ax.set_ylabel('$M(kR)$',fontsize=25)
    ax.set_xlabel('$kR$',fontsize=25)
    # Return output
    return fig, ax

#%% HISTOGRAM_DATA plots

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
    histogram_data: HISTOGRAM_DATA
        The object that contains the histogram data from the simulation.
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
        # Normalize distribution
        r_array = histogram_data.y_bins # dimensionless
        f_array /= intg.simpson(f_array, x=r_array) # dimensionless
        # Plot!
        ax.bar(edges[:-1],f_array,width=np.diff(edges),**kwargs)
        # Prettify plot
        ax.set_ylabel('Radial distribution',fontsize=25)
        ax.set_xlabel('$y\\equiv r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif histogram_data.quantity == 'velocity':
        # Normalize distribution
        v_array = histogram_data.y_bins # dimensionless
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
    histogram_data: HISTOGRAM_DATA
        The object that contains the histogram data from the simulation.
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
            (alpha, beta), y_array = histogram_data.fit_histogram(z_em=z_em)
        else:
            (mu, sigma), y_array = histogram_data.fit_histogram(z_em=z_em)
    else:
        if histogram_data.quantity == 'distance':
            (alpha, beta), y_array = histogram_data.fit_histogram(x_em=x_em)
        else:
            (mu, sigma), y_array = histogram_data.fit_histogram(x_em=x_em)
    # Get more samples to draw nice functions
    if histogram_data.quantity == 'distance':
        y_array_new = np.linspace(0,1,1000)
    else:
        y_array_new = np.linspace(y_array[0],y_array[-1],1000)
    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    # Plot the analytical fit
    if histogram_data.quantity == 'distance':
        ax.plot(y_array_new, beta_dist.pdf(y_array_new,alpha,beta),**kwargs)
        # Prettify plot
        ax.set_ylabel('Radial distribution',fontsize=25)
        ax.set_xlabel('$y\\equiv r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
    elif histogram_data.quantity == 'velocity':
        ax.plot(y_array_new, norm_dist.pdf(y_array_new,mu,sigma), **kwargs)
        ax.set_ylabel('Velocity distribution',fontsize=25)
        ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=20)
    # Return output
    return fig, ax