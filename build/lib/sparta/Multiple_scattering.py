"""
This is the main script for studying the scattered trajectories of
Lyman alpha photons in the intergalactic medium (IGM). 
The script works on a virtual "grid" and backwards in time, i.e. 
we begin from Lyman alpha frequency and track the photon's past trajectory. 
This is done that way because we can interepret each scattering point as a 
potential source of emission.
At each "cell" of the grid the commulative optical depth is updated, and 
once it crosses a random threshold (drawn from exp(-tau) distribution), 
the photon is scattered to a random direction (from an isotropic/unisotropic
angular distribution). 
The optical depth depends both on the temperature (a free parameter in the
simulation) and the apparent frequency that hydrogen atom in the IGM sees.
The scripts accounts for the redshift of the photon (or blueshift, since
we work backwards), and also for the bulk peculiar motion of gas particles
in the IGM. This is done by correlating the smoothed velocities at 
neighbouring cells and computing the projected relative velocity along the
photon's trajectory.
The script allows simulating an arbitrary number of photons, each of which
has its own special random seed. Once the simulation is over, the user can
plot the evolution of individual photons by tracking their progress in 
redshift, apparent frequency, and distance from the absorber. In addition,
scatter plots of the redshift of the emitter vs. the distance the photon has 
traveled (or vs. the relative velocity of the abosrber to the emitter) can
be plotted.  
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
import tqdm
from classy import Class
import scipy.integrate as intg
from numpy.linalg import norm
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.stats import norm as norm_dist
from scipy.stats import beta as beta_dist


plt.rcParams.update({"text.usetex": True,
                     "font.family": "Times new roman"})
colors =  ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

#%% Define some global parameters
Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
h_P = 6.62606896e-34 # Planck Constant in J*sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
lambda_alpha = c/nu_Lya # Lya wavelength in m
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
Tcmb0 = 2.728 # CMB temperature in Kelvin
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya
Lyman_beta = 32./27. # Lyb frequency in units of Lya frequency

#%% Define some helpful functions for the computation

def compute_Lya_cross_section(nu_apparent,T,flag='LORENTZIAN'):
    """
    Compute the Lyman alpha cross section for scattering a photon with 
    a given frequency with IGM hydrogen atoms with given temperature.
    
    Parameters
    ----------
    nu_apparent: float
        The dimensionless frequency of the photon (in units of Lyman alpha frequency).
    T: float
        Temperature of the IGM (in Kelvin degrees)
    flag: string, optional
        Type of cross-section to be used. Options are (default is 'Lorenzian'):
            - 'LORENTZIAN': A simple Lorentzian profile is used when T=0. 
            For a finite temperature, the Lorentzian profile is convolved 
            with a Gaussian to yield to Voigt profile.
            - 'PEEBLES': Cross-section from Peebles, P.J.E. (1993) Principles 
            of Physical Cosmology (see also Eq. 49 in arXiv: 1704.03416).
            Only works for T=0.
    
    Returns
    -------
    float:
        Cross-section in m^2.
    """
    # If T=0, we can use either Lorentzian or Peebles profiles
    if T == 0.:
        sigma_Lya = 3.*lambda_alpha**2/(32.*np.pi**3) # m^2
        if flag == 'LORENTZIAN':
            sigma_Lya *= A_alpha_dimensionless**2 # m^2
            sigma_Lya /= (nu_apparent-1.)**2 + A_alpha_dimensionless**2 /16./np.pi**2 # m^2
        elif flag == 'PEEBLES':
            sigma_Lya *= A_alpha_dimensionless**2 * nu_apparent**4 # m^2
            sigma_Lya /= (nu_apparent-1.)**2 + A_alpha_dimensionless**2 * nu_apparent**6 /16./np.pi**2 # m^2
    # Otherwise, we use the Voigt profile
    else:
        # Assuming neutral IGM (m_b = 1.22*m_H)
        Delta_nu = np.sqrt(2*k_B*T/1.22/m_H/c**2) # dimensionless (in units of nu_Lya)
        a = A_alpha_dimensionless/4/np.pi/Delta_nu # dimensionless
        x = (nu_apparent-1.)/Delta_nu # dimensionless
        sigma_Lya = (3.*lambda_alpha**2 /2.)*a * voigt_profile(x,1./np.sqrt(2.),a) # m^2
    # Return output
    return sigma_Lya

def calculate_rotation_matrix(mu,phi):
    """
    Construct a rotation matrix that rotates the third axis by mu=cos(theta)
    and then the first axis by phi.
    
    Parameters
    ----------
    mu: float
        The cosine of the angle in which the third axis is rotated.
    phi: float
        The angle in which the first axis is rotated (in radians).
    
    Returns
    -------
    numpy array (3X3):
        The rotation matrix.
    """
    rotation_phi = np.array([
                            [1.,                0.,                  0.],
                            [0.,       np.cos(phi),        -np.sin(phi)],
                            [0.,       np.sin(phi),         np.cos(phi)]
                            ])
    rotation_mu = np.array([
                           [mu,                  -np.sqrt(1.-mu**2),   0.],
                           [np.sqrt(1.-mu**2),    mu,                  0.],
                           [0.,                   0.,                  1.]
                           ])
    # Return output
    return rotation_phi.dot(rotation_mu)

def inverse_mu_CDF(a,b):
    """
    Construct an interpolation table for the inverse CDF that is associated
    with the PDF a*+b*mu^2, where mu ranges from -1 to +1.
    
    Parameters
    ----------
    a, b: floats
        The parameters of the PDF.
    
    Returns
    -------
    :class: 'scipy.interpolate._interpolate.interp1d':
        The interpolation of the inverse CDF.
    """
    # Sanity check: the distribution has to be normalized to 1
    if not 2.*(a+b/3.) == 1.:
        print("Warning: mu distribution is not normalized to 1")
    mu_arr = np.linspace(-1.,1.,100)
    CDF_arr = a*(mu_arr + 1.) + b/3.*(mu_arr**3 + 1.)
    # Return output
    return interp1d(CDF_arr,mu_arr,kind='cubic')
    
def detetermine_u0(x,a):
    """
    Determine u0, the value that minimizes the fraction of generated values 
    that will be discarded. We follow Eq. (22) in arXiv: 2001.11252.
    
    Parameters
    ----------
    x: float
        The frequency distance from Lyman alpha, in units of line's width,
        x = (nu - nu_Lya)/Delta_nu.
    a: float
        The ratio of natural to thermal line broadening, a = A_alpha/(4*pi*Delta_nu).
    
    Returns
    -------
    float:
        The value of u0.
    """
    
    zeta = np.log10(a)
    u0 = (2.648963 + 2.014446*zeta + 0.351479*zeta**2
        + x*(-4.058673 - 3.675859*zeta - 0.640003*zeta**2
        + x*(3.017395 + 2.117133*zeta + 0.370294*zeta**2
        + x*(-0.869789 - 0.565886*zeta - 0.096312*zeta**2
        + x*(0.110987 + 0.070103*zeta + 0.011557*zeta**2
        + x*(-0.005200 - 0.003240*zeta - 0.000519*zeta**2
        ))))))
    # Return output
    return u0

def draw_from_voigt_distribution(x,a):
    """
    Draw a random variable from the "Voigt" distribution,
    ~ exp(-u^2)/((u-x)^2+a^2).
    Note: This is actually NOT the Voigt distribution, as the
    argument of the distribution is u. The Voigt distribution is 
    achieved by integrating over the above distribution and it is
    a function of x, not u.
    
    Parameters
    ----------
    x: float
        The frequency distance from Lyman alpha, in units of line's width,
        x = (nu - nu_Lya)/Delta_nu.
    a: float
        The ratio of natural to thermal line broadening, a = A_alpha/(4*pi*Delta_nu).
    
    Returns
    -------
    float:
        A random variable.
    """
    
    # Below we follow what is implemented in the RASCAS code (arXiv: 2001.11252,
    # section 3.3.3)
    if x < 8.:
        # Determine u0, the value that minimizes the fraction of generated values 
        # that will be discarded
        u0 = detetermine_u0(x,a)
        # A is the missing factor of proportionality in Eq. (21), determined 
        # such that the integral over g(u) is unity 
        A = np.pi/2.*(1.+np.exp(-u0**2)) + (1.-np.exp(-u0**2))*np.arctan((u0-x)/a)
        # Draw 2 random variables from a uniform distribution between 0 and 1
        Y1, Y2 = np.random.rand(2)
        # If Y1 (CDF of g(u)) is small, we compute u from the inverse CDF of g1(u) 
        if Y1 < (np.arctan((u0-x)/a) + np.pi/2.)/A:
            u = x + a*np.tan(Y1*A-np.pi/2.)
            # Reject some of the samples if Y2 is large enough and try again
            if Y2 >= np.exp(-u**2):
                try:
                    u = draw_from_voigt_distribution(x,a)
                except RecursionError:
                    pass
        # Otherwise, we compute u from the inverse CDF of g2(u)
        else:
            u = x + a*np.tan(np.exp(u0**2)*(Y1*A-(1.-np.exp(-u0**2))*np.arctan((u0-x)/a)-np.pi/2.))
            # Reject some of the samples if Y2 is large enough and try again
            if Y2 >= np.exp(-u**2)/np.exp(-u0**2):
                try:
                    u = draw_from_voigt_distribution(x,a)
                except RecursionError:
                    pass
    # If x >= 8, we draw u from a normal distribution with mean 1/x and variance 1/2
    else:
        u = np.random.normal(loc=1./x,scale=1./np.sqrt(2.))
    # Return output
    return u

def correlation_function(r,cosmo_params,kind='density',normalization=True):
    """
    Compute the 2-point correlation at z=0 as a function of 
    comoving distance.
    
    Parameters
    ----------
    r: float
        The comoving distance in Mpc.
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    kind: str
        Kind of correlation function to plot: options are 'density', 
        'v_parallel' and 'v_perp'.
    normalization: bool
        Whether to normalize the correlation function with the RMS.
    
    Returns
    -------
    float:
        The 2-point correlation.
    """
    
    # Extract transfer function of matter density at z=0
    k_CLASS = cosmo_params.CLASS_OUTPUT.get_transfer(z=0.)['k (h/Mpc)'][:]*cosmo_params.h # 1/Mpc
    Transfer_z0 = cosmo_params.CLASS_OUTPUT.get_transfer(z=0.)
    if kind == 'density':
        transfer_CLASS = Transfer_z0['d_tot'][:]
    elif kind == 'v_parallel' or  kind == 'v_perp':
        transfer_CLASS = Transfer_z0['t_b'][:]/k_CLASS/np.sqrt(3.) # dimensionless
    else:
        raise ValueError("'kind' can only be 'density', 'v_parallel'  or 'v_perp'.")

    # Interpolate transfer function
    if r > 0:
        k_arr = np.logspace(-4.,4.,10000)/r
        k_arr = np.concatenate((np.linspace(min(k_CLASS),min(k_arr),100),k_arr))
        k_arr = np.unique(k_arr)
        k_arr = np.sort(k_arr)
    else:
        k_arr = k_CLASS
    transfer = interp1d(k_CLASS, transfer_CLASS,
                        kind='cubic',bounds_error=False,fill_value=0.)(k_arr)
    # Power spectrum of primordial curvature fluctuations
    P_R = cosmo_params.A_s*pow(k_arr/0.05,cosmo_params.n_s-1.)
    # Window function
    kr = k_arr*r # dimensionless
    with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
        if kind == 'density':
            W_k =  np.sin(kr)/kr # dimensionless
        elif kind == 'v_parallel':
            W_k = (3.*(kr**2-2.)*np.sin(kr)+6.*kr*np.cos(kr))/kr**3 # dimensionless
        elif kind == 'v_perp':
            W_k = 3.*(np.sin(kr)-kr*np.cos(kr))/kr**3 # dimensionless
    # Taylor expansion for small kr
    kr_small = kr[kr < 1.e-3]
    if kind == 'density':
        W_k[kr < 1.e-3] = 1.-(kr_small**2)/6.
    elif kind == 'v_parallel':
        W_k[kr < 1.e-3]= 1.-3.*(kr_small**2)/10.
    elif kind == 'v_perp':
        W_k[kr < 1.e-3] = 1.-(kr_small**2)/10.
    # Integrate to get the correlation function xi(r)
    integrand = P_R*transfer*transfer*W_k/k_arr # Mpc
    xi = intg.simpson(integrand, x=k_arr) # dimensionless
    # Normalize if needed
    if normalization:
        xi /= intg.simpson(P_R*transfer*transfer/k_arr, x=k_arr) # dimensionless
    # Return output
    return xi

#%% Define some helpful functions for plotting (with no data from simulation)

def plot_cross_section(T=0.,
                       nu_min=None,
                       nu_max=1.e-3,
                       velocity=0.,
                       flag='LORENTZIAN',
                       ax=None,
                       **kwargs):
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

def plot_delta_tau(Delta_L,
                   cosmo_params,
                   T=0.,
                   nu_lim=1.e-3,
                   velocity=0.,
                   z=20.,
                   flag='LORENTZIAN',
                   ax=None,
                   **kwargs):
    """
    Plot differential optical depth as a function of frequency
    for a photon traveling a cell size.

    Parameters
    ----------
    
    Delta_L: float
        The cell size of the simulation in Mpc.
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    T: float, optional
        Temperature of the IGM (in Kelvin degrees)
    nu_lim: float, optional
        Range of frequencies for which the cross-section will be displayed,
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

    nu_array_zero = 1.+np.linspace(-nu_lim,nu_lim,1000)
    nu_array = nu_array_zero/(1.-velocity)
    n_HI = cosmo_params.n_H_z0*(1.+z)**3
    sigma_array = np.array([compute_Lya_cross_section(nu,T,flag=flag) for nu in nu_array])
    dtau_array = n_HI*sigma_array/(1.+z)* Delta_L * Mpc_to_meter # dimensionless
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    ax.semilogy(nu_array_zero-1.,dtau_array,**kwargs)
    ax.set_xlabel('$\\nu/\\nu_\\alpha-1$',fontsize=25)
    ax.set_ylabel('$\\Delta\\tau$',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if "label" in kwargs:
        ax.legend(fontsize=15)
    # Return output
    return fig, ax

#%% Class for setting simulation parameters

class SIM_PARAMS():
    """
    Class for setting simulation parameters.
    
    Parameters
    ----------
    sim_params: dictionary
        Contains the following simulation parameters:
            - z_abs: float
                The redshift where the photon was absorbed.
            - N: int 
                Number of photons to simulate.
            - Delta_L: float
                "Grid" resolution in Mpc.
            - x_stop: float
                Distance from absorption point to stop the simulation in units of the diffusion scale.
            - INCLUDE_VELOCITIES: bool
                If True, bulk peculiar velocities will be included in the simulation.
                Otherwise, assumes zero velocity throughout the simulation.
            - NO_CORRELATIONS: bool
                If True, the correlations in the velocitiy field will be ignored and at each sample a
                random velocity vector will be drawn independently from the previous velocity vector.
            - USE_INTERPOLATION_TABLES: bool
                If True, interpolation tables for velocity rms and correlation coefficients will be used,
                Otherwise, the rms and correlation coefficients will be computed from their integral definitions
                every time they are needed (MUCH more slowly that way).
            - INCLUDE_TEMPERATUR:, bool
                If True, non-zero Temperature will be accounted, based on its value in cosmo_params.
                Otherwise, zero temperature is assumed.
            - ANISOTROPIC_SCATTERING: bool
                If True, mu is drawn from a non-uniformal distribution (see Eq. 20 in arXiv: 2311.03447).
                Otherwise, it is drawn from a uniformal distribution.
            - INCLUDE_RECOIL: bool
                If True, atom recoil will be included during a scattering event (see Eq. 23 in arXiv: 2311.03447).
                Otherwise, this effect is not included and the outgoing photon's frequency is the same as 
                the incoming photon's frequency, in the atom rest frame.
            - STRAIGHT_LINE: bool
                If True, no scattering will happen and the photon will travel in a straight line.
                Otherwise, scattering are allowed.
            - CROSS_SECTION: string
                Type of cross-section to be used. Options are (default is 'Lorenzian'):
                    - 'LORENTZIAN': A simple Lorentzian profile is used when T=0. 
                    For a finite temperature, the Lorentzian profile is convolved 
                    with a Gaussian to yield to Voigt profile.
                    - 'PEEBLES': Cross-section from Peebles, P.J.E. (1993) Principles 
                    of Physical Cosmology (see also Eq. 49 in arXiv: 1704.03416).
                    Only works for T=0.
    
    """
    
    def __init__(self,sim_params):
        for k, v in sim_params.items():
            setattr(self, k, v)
        if not "x_stop" in  sim_params.keys():
            self.x_stop = None

#%% Class for setting cosmological parameters 
#   (and other variables/function with respect to the cosmological parameters)

class COSMO_PARAMS():
    """
    Class for setting cosmological parameters.
    
    Parameters
    ----------
    cosmo_params: dictionary
        Contains the following cosmological parameters:
            - h: float
                Hubble constant (in 100 km/sec/Mpc).
            - Omega_m: float
                Current matter density parameter.
            - Omega_b: float
                Current baryon density parameter.
            - A_s: float
                Amplitude of primordial curvator fluctuations 
                in the pivot scale 0.05 Mpc^-1.
            - n_s: float
                Spectral tilt of primordial curvator fluctuations.
            - T: float
                Temperature of the IGM in the simulation, in Kelvin.
            - x_HI: float
                Fraction of neutral hydrogen in the simulation.
    """
    
    def __init__(self,cosmo_params):
        for k, v in cosmo_params.items():
            setattr(self, k, v)
            
        # Compute other cosmological parameters
        self.H0 = 100*self.h # Hubble constant in km/sec/Mpc
        self.Omega_c = self.Omega_m - self.Omega_b # CDM portion
        self.rho_crit = 1.8788e-26 * self.h**2 # Critical energy density in kg/m^3
        self.rho_b0 = self.rho_crit*self.Omega_b # Baryon energy density in kg/m^3
    
    def run_CLASS(self):
        """
        Run CLASS with the input cosmological parameters.
        
        """
        CLASS_params = {}
        CLASS_params['h'] = self.h
        CLASS_params['Omega_cdm'] = self.Omega_c
        CLASS_params['Omega_b'] = self.Omega_b
        CLASS_params['A_s'] = self.A_s
        CLASS_params['n_s'] = self.n_s
        CLASS_params['T_cmb'] = Tcmb0
        CLASS_params['m_ncdm'] = "0.06" # eV
        CLASS_params['N_ncdm'] = 1
        CLASS_params['N_ur'] = 2.0308
        CLASS_params['output'] = 'mTk,vTk'
        CLASS_params['z_pk'] = 60.
        CLASS_params['gauge']='Newtonian'
        CLASS_params['P_k_max_1/Mpc'] = 100. # A rather high value for computing correlation at small distances
        # Run CLASS
        CLASS_OUTPUT = Class()
        CLASS_OUTPUT.set(CLASS_params)
        CLASS_OUTPUT.compute()
        self.CLASS_OUTPUT = CLASS_OUTPUT
        # Helium mass fraction (rho_He/rho_b)
        self.YHe = CLASS_OUTPUT.get_current_derived_parameters(['YHe'])['YHe']
        # Hydrogen number density at z=0
        self.n_H_z0 = (1.-self.YHe)*self.rho_b0/m_H # m^-3
        # Mean baryon mass. It is assumed that helium is doubly ionized when hydrogen is ionized
        self.m_b = m_H * (1./((2.-self.x_HI)*(1.-self.YHe)+(1.+2.*(1.-self.x_HI))*self.YHe/4.)) # kg
        
    def Delta_nu_star(self,z_abs):
        """
        Compute the frequency shift (in units of Lyman alpha frequncy) between 
        the absorption point and the point in which a photon that is escaping
        to infinity sees an optical depth of 1.
        
        Parameters
        ----------
        z_abs: float
            Redshift of absorption.
        
        Returns
        -------
        float:
            The frequency shift Delta_nu_star as given by Eq. (8) 
            in arXiv: astro-ph/9902180 (assuming matter domination). 
        """
        
        nu_star = 3.*c**3*A_alpha**2*self.n_H_z0*self.x_HI*pow(1.+z_abs,3./2.) # Hz^5
        nu_star /= 32.*np.pi**3 *nu_Lya**4 * (1000.*self.H0/Mpc_to_meter)*np.sqrt(self.Omega_m) # dimensionless
        return nu_star
    
    def R_SL(self,z1,z2):
        """
        Compute comoving straight line distance between z1 and z2.
        
        Parameters
        ----------
        z1: float
            Initial redshift.
        z2: float
            Final redshift.
        
        Returns
        -------
        float:
            The comoving distance between z1 and z2, in Mpc.
            Based on Eq. A3 in arXiv: 2101.01777 (This is int{c/H(z)} while
            assuming matter domination). 
        """
        
        r = abs(2.*c/1000./(self.H0*np.sqrt(self.Omega_m))*(1./np.sqrt(1+z1)-1./np.sqrt(1+z2))) # Mpc
        return r

    def R_SL_inverse(self,z1,r):
        """
        Compute the redshift that is located a comoving distance r fron z1
        (assuming z2 > z1).
        
        Parameters
        ----------
        z1: float
            Initial redshift.
        r: float
            Comoving distance, in Mpc.
        
        Returns
        -------
        float:
            The next redshift. 
        """
        z2 = pow(1./np.sqrt(1+z1)-1000.*self.H0*np.sqrt(self.Omega_m)*r/(2.*c),-2)-1. # Mpc
        return z2
    
    def r_star(self,z_abs):
        """
        Compute the comoving diffustion scale at z_abs.
        
        Parameters
        ----------
        z_abs: float
            Redshift of absorption.
        
        Returns
        -------
        float:
            The comoving diffusion scale in units of Mpc. This corrsponds to 
            the comoving distance that the photon has to travel (backwards in
            time) from the absorption point to the point from which the photon
            sees an optical depth of unity when it escapes to infinity. 
        """
        
        # What we have below is actually (z_star-z_abs)/(1+z_abs) = (nu_star-nu_Lya)/nu_Lya
        Delta_z_star = self.Delta_nu_star(z_abs) # dimensionless
        # Return output
        return self.R_SL(z_abs,z_abs + (1.+z_abs)*Delta_z_star) # Mpc 

    def compute_RMS(self,z,r):
        """
        Compute 1D velocity rms at redshift z, smoothed on a scale r.
        
        Parameters
        ----------
        z: float
            Redshift.
        r: float
            Smoothing radius, in Mpc.
        
        Returns
        -------
        float:
            The smoothed 1D velocity RMS, given in units of c. 
        """
    
        # Extract transfer function
        k_CLASS = self.CLASS_OUTPUT.get_transfer(z=z)['k (h/Mpc)'][:]*self.h # 1/Mpc
        Transfer_z = self.CLASS_OUTPUT.get_transfer(z=z)
        theta_b_z_CLASS = Transfer_z['t_b'][:] # 1/Mpc
        # Interpolate transfer function
        k_arr = np.logspace(-4.,4.,10000)/r
        k_arr = np.concatenate((np.linspace(min(k_CLASS),min(k_arr),100),k_arr))
        k_arr = np.unique(k_arr)
        k_arr = np.sort(k_arr) # 1/Mpc
        theta_b_z = interp1d(k_CLASS, theta_b_z_CLASS,
                             kind='cubic',bounds_error=False,fill_value=0.)(k_arr) # 1/Mpc
        # Define velocity transfer function (division by sqrt(3) is because we want 1D transfer function)
        v_b_z = theta_b_z/k_arr/np.sqrt(3.) # dimensionless
        # Power spectrum of primordial curvature fluctuations
        Delta_R_sq = self.A_s*pow(k_arr/0.05,self.n_s-1.) # dimensionless
        # Window function (for smoothing the fields at scale r)
        kr = k_arr*r # dimensionless
        with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
            W_k_top_hat = 3.*(np.sin(kr)-kr*np.cos(kr))/kr**3 # dimensionless
        # Taylor expansion for small kr
        kr_small = kr[kr < 1.e-3]
        W_k_top_hat[kr < 1.e-3] = 1.-(kr_small**2)/10.
        # Multiply transfer function by window function to smooth the fiedls at scale r
        # WARNING: this smoothing assumes that r is of the cell size!
        v_b_z *= W_k_top_hat # dimensionless
        # Integrate to get the variances
        variance_velocity = intg.simpson(v_b_z**2 * Delta_R_sq /k_arr, x=k_arr) # dimensionless
        # Return output
        # We take the square root as we want the rms, not the variances.
        return np.sqrt(variance_velocity) # dimensionless
    
    def compute_2_point_correlation(self,z1,z2,v1_1D_rms,v2_1D_rms,r=None):
        """
        Compute velocity correlation coefficients for parallel 
        and perpendicular components, between two points at redshifts 
        z1 and z2.
        
        Parameters
        ----------
        z1: float
            Initial redshift.
        z2: float
            Final redshift.
        v1_1D_rms: float
            1D velocity rms at z1, in units of c.
        v2_1D_rms: float
            1D velocity rms at z2, in units of c.
        r: float, optional
            Comoving distance between the two points, in Mpc.
        
        Returns
        -------
        rho_v_parallel: float
            Correlation coefficient for the parallel component of the velocity. 
        rho_v_perp: float
            Correlation coefficient for the perpendicular component of the velocity. 
        """
        # Extract transfer functions
        k_CLASS = self.CLASS_OUTPUT.get_transfer(z=z1)['k (h/Mpc)'][:]*self.h # 1/Mpc
        Transfer_z1 = self.CLASS_OUTPUT.get_transfer(z=z1)
        Transfer_z2 = self.CLASS_OUTPUT.get_transfer(z=z2)
        theta_b_z1_CLASS = Transfer_z1['t_b'][:] # 1/Mpc
        theta_b_z2_CLASS = Transfer_z2['t_b'][:] # 1/Mpc
        # Comoving distance between z1 and z2
        if r is None:
            r = self.R_SL(z1,z2) # Mpc
        # Interpolate transfer functions
        k_arr = np.logspace(-4.,4.,10000)/r
        k_arr = np.concatenate((np.linspace(min(k_CLASS),min(k_arr),100),k_arr))
        k_arr = np.unique(k_arr)
        k_arr = np.sort(k_arr) # 1/Mpc
        theta_b_z1 = interp1d(k_CLASS, theta_b_z1_CLASS,
                             kind='cubic',bounds_error=False,fill_value=0.)(k_arr) # 1/sec
        theta_b_z2 = interp1d(k_CLASS, theta_b_z2_CLASS,
                             kind='cubic',bounds_error=False,fill_value=0.)(k_arr) # 1/sec
        # Define velocity transfer functions (division by sqrt(3) is because we want 1D transfer functions)
        v_b_z1 = theta_b_z1/k_arr/np.sqrt(3.) # dimensionless
        v_b_z2 = theta_b_z2/k_arr/np.sqrt(3.) # dimensionless
        # Window functions
        kr = k_arr*r # dimensionless
        with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
            W_k_top_hat = 3.*(np.sin(kr)-kr*np.cos(kr))/kr**3 # dimensionless
            W_k_parallel = (3.*(kr**2-2.)*np.sin(kr)+6.*kr*np.cos(kr))/kr**3 # dimensionless
            W_k_perp = 3.*(np.sin(kr)-kr*np.cos(kr))/kr**3 # dimensionless
        # Taylor expansion for small kr
        kr_small = kr[kr < 1.e-3]
        W_k_top_hat[kr < 1.e-3] = 1.-(kr_small**2)/10.
        W_k_parallel[kr < 1.e-3]= 1.-3.*(kr_small**2)/10.
        W_k_perp[kr < 1.e-3] = 1.-(kr_small**2)/10.
        # Multiply transfer functions by window function to smooth the fiedls at scale r.
        # WARNING: this smoothing assumes that r is of the cell size!
        v_b_z1 *= W_k_top_hat # dimensionless
        v_b_z2 *= W_k_top_hat # dimensionless
        # Power spectrum of primordial curvature fluctuations
        Delta_R_sq = self.A_s*pow(k_arr/0.05,self.n_s-1.)
        # Calculate correlation coefficients
        rho_v_parallel = intg.simpson(v_b_z1*v_b_z2* Delta_R_sq * W_k_parallel/ k_arr,x=k_arr) # dimensionless
        rho_v_perp = intg.simpson(v_b_z1*v_b_z2* Delta_R_sq * W_k_perp/ k_arr,x=k_arr) # dimensionless
        rho_v_parallel /= v1_1D_rms*v2_1D_rms # dimensionless
        rho_v_perp /= v1_1D_rms*v2_1D_rms # dimensionless
        # Sanity check: -1 <= rho <= 1
        if rho_v_parallel**2 > 1:
            print(f"Warning: At (z1,z2)={z1,z2} the correlation coefficient for v_parallel is rho={rho_v_parallel}")
        if rho_v_perp**2 > 1:
            print(f"Warning: At (z1,z2)={z1,z2} the correlation coefficient for v_perp is rho={rho_v_parallel}")
        # Return output
        return rho_v_parallel, rho_v_perp

#%% Class for managing a cosmological point in spacetime in which the photon has passed

class COSMO_POINT_DATA():
    """
    Class for managing a cosmological point in spacetime in which the photon 
    has passed.
    
    Parameters
    ----------
    redshift: float
        Redshift of the point.
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    sim_params: :class:`~SIM_PARAMS`
        The simulation parameters.
    velocity_vector: numpy array (3X1), optional
        Bulk peculiar velocity vector at the point, in units of c.
            - The first component is parallel to the photon trajectory (n_||)
            - The second component is perpendicular to the photon trajectory (n_perp)
            - The third component is perpendicular to the above (n_cross)
    position_vector: numpy array (3X1), optional
        Position vector of the point, with respect to absorption point, 
        in units of Mpc. Unlike the velocity vector, the components of the 
        position vector are given in a fixed comoving frame.
    rotation_matrix: numpy array (3X3), optional
        The total rotation matrix who's inverse can be used to transform
        the velocity vector in z_abs to the same coordinate system of the point.
    apparent_frequency: float, optional
        The dimensionless frequency of the photon (in units of Lyman alpha frequency).
    velocity_1D_rms: float, optional
        The smoothed 1D velocity RMS, given in units of c.
    """
    
    def __init__(self,
                 redshift,
                 cosmo_params,
                 sim_params,
                 velocity_vector = None,
                 position_vector = np.zeros(3),
                 rotation_matrix = np.eye(3),
                 apparent_frequency = None,
                 velocity_1D_rms = None
    ):
        self.redshift = redshift
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.velocity_vector = velocity_vector
        self.position_vector = position_vector
        self.rotation_matrix = rotation_matrix
        self.apparent_frequency = apparent_frequency
        self.velocity_1D_rms = velocity_1D_rms
    
    def copy(self):
        """
        Copy the content of this point to another object.
        
        Returns
        -------
        :class:`~COSMO_POINT_DATA`
            A copy of the content of this point
        """
        
        if self.sim_params.INCLUDE_VELOCITIES:
            velocity_vector = self.velocity_vector.copy(),
            rotation_matrix = self.rotation_matrix.copy(),
            velocity_1D_rms = self.velocity_1D_rms
            # This is weird, for some reason those variables are a
            # tuple of a single element. I'm making it an array below
            velocity_vector = velocity_vector[0]
            rotation_matrix = rotation_matrix[0]
        else:
            velocity_vector = None,
            rotation_matrix = None,
            velocity_1D_rms = None

        # Return output
        return COSMO_POINT_DATA(redshift=self.redshift,
                                cosmo_params=self.cosmo_params,
                                sim_params=self.sim_params,
                                velocity_vector=velocity_vector,
                                position_vector=self.position_vector.copy(),
                                rotation_matrix=rotation_matrix,
                                apparent_frequency=self.apparent_frequency,
                                velocity_1D_rms=velocity_1D_rms
                                )
    
    def rotate(self,mu,phi):
        """
        Rotate the coordinate system of the point by mu=cos(theta) and phi. 
        
        Parameters
        ----------
        mu: float
            The cosine of the angle in which the third axis is rotated.
        phi: float
            The angle in which the first axis is rotated (in radians).
        """
        R_matrix = calculate_rotation_matrix(mu,phi)
        R_matrix_inv = np.linalg.inv(R_matrix)
        # Update rotation matrix
        self.rotation_matrix = self.rotation_matrix.dot(R_matrix)
        # Rotate velocity vector
        # Note: why do I rotate by the inverse matrix? This is because:
        #   (1) The velocity vector is rotated passively (the vector remains the same,
        #       it is the coordinate system that was transformed in order to align the
        #       first component with the photon's new direction), 
        #       while the displacement vector (see update_position below) is rotated 
        #       actively (it is the dispalcement vector that is rotating, not the 
        #       coordinate system) in order to bring the photon to its correct position.
        #   (2) The n'th rotation matrix for the n'th point is R_tot = R1*R2*...*Rn.
        #       Thus, the inverse matrix is R_tot_inv = Rn_inv*...R1_inv.
        #       At the n'th point, the velocity vector is rotated by R_tot_inv compared
        #       to the fixed grid's frame. Thus, we need to multiply it by R_tot
        #       in order to compare it with the velocity vector at z_abs.
        #       This is exactly what we do when we collect the data in SIM_DATA() below.
        self.velocity_vector = R_matrix_inv.dot(self.velocity_vector) # dimensionless
    
    def compute_RMS(self):
        """
        Compute the smoothed 1D velocity rms for this point. 
        Interpolation is performed if USE_INTERPOLATION_TABLES = True.
        """
        # Interpolate!
        if self.sim_params.USE_INTERPOLATION_TABLES:
            try:
                self.velocity_1D_rms = self.cosmo_params.interpolate_RMS(self.redshift)
            except ValueError:
                self.velocity_1D_rms = self.cosmo_params.interpolate_RMS(self.cosmo_params.redshift_grid[-1])
        # Integrate!
        else:
            self.velocity_1D_rms = self.cosmo_params.compute_RMS(z=self.redshift,
                                                                 r=self.sim_params.Delta_L)
        
    def compute_2_point_correlation(self,z1_data,r=None):
        """
        Compute velocity correlation coefficients for parallel and 
        perpendicular components, between the current point and the one 
        defined by z1_data. Interpolation is performed if 
        USE_INTERPOLATION_TABLES = True.
        
        Parameters
        ----------
        z1_data: :class:`~COSMO_POINT_DATA`
            The data of the previous point.
        r: float, optional
            Comoving distance between the two points, in Mpc.
        """
        if not self.sim_params.NO_CORRELATIONS:
            # Interpolate!
            if self.sim_params.USE_INTERPOLATION_TABLES:
                self.rho_v_parallel = self.cosmo_params.interpolate_rho_parallel(self.redshift,z1_data.redshift)[0,0]
                self.rho_v_perp = self.cosmo_params.interpolate_rho_perp(self.redshift,z1_data.redshift)[0,0]
                # Sanity check: -1 <= rho <= 1
                if self.rho_v_parallel**2 > 1.:
                    print(f"Warning: At (z1,z2)={z1_data.redshift,self.redshift} the correlation coefficient for v_parallel is rho={self.rho_v_parallel}")
                if self.rho_v_perp**2 > 1.:
                    print(f"Warning: At (z1,z2)={z1_data.redshift,self.redshift} the correlation coefficient for v_perp is rho={self.rho_v_perp}")
            # Integrate!
            else:
                self.rho_v_parallel, self.rho_v_perp = self.cosmo_params.compute_2_point_correlation(z1=z1_data.redshift,
                                                                                                    z2=self.redshift,
                                                                                                    v1_1D_rms=z1_data.velocity_1D_rms,
                                                                                                    v2_1D_rms=self.velocity_1D_rms)
        else:
            self.rho_v_parallel = 0.
            self.rho_v_perp = 0.
                
    
    def draw_conditional_velocity_vector(self,z1_data):
        """
        Draw a conditional velocity vector for this point based on the 
        velocity vector of the previous sample.
        
        Parameters
        ----------
        z1_data: :class:`~COSMO_POINT_DATA`
            The data of the previous point.
        
        """
        # Compute the 2-point correlation coefficients for the parallel and 
        # perpendicular components of the velocity field.
        self.compute_2_point_correlation(z1_data)
        # Comopute the conditional mean and variance for the current velocity
        # components, based on the previous point.
        # For two Gaussian random variables X and Y, the conditional mean (mu) 
        # and variance (sigma^2) of X given Y=y are
        #
        #                   mu_X|Y = mu_X + rho_XY*sigma_X/sigma_Y*(y-mu_y)
        #
        #                   sigma_X|Y = sigma_X*sqrt(1-rho_XY^2)
        #
        # This is what the following lines do (in our case, mu_X=mu_Y=0)
        mu_parallel = (self.velocity_1D_rms/z1_data.velocity_1D_rms
                       *self.rho_v_parallel*z1_data.velocity_vector[0]) # dimensionless
        mu_perp = (self.velocity_1D_rms/z1_data.velocity_1D_rms
                       *self.rho_v_perp*z1_data.velocity_vector[1]) # dimensionless
        mu_cross = (self.velocity_1D_rms/z1_data.velocity_1D_rms
                       *self.rho_v_perp*z1_data.velocity_vector[2]) # dimensionless
        sigma_parallel = self.velocity_1D_rms*np.sqrt(1.-self.rho_v_parallel**2) # dimensionless
        sigma_perp = self.velocity_1D_rms*np.sqrt(1.-self.rho_v_perp**2) # dimensionless
        sigma_cross = self.velocity_1D_rms*np.sqrt(1.-self.rho_v_perp**2) # dimensionless
        # Draw a normal random velocity vector with the appropriate mean 
        # and variance for all components
        self.velocity_vector = np.zeros(3)
        self.velocity_vector[0] = np.random.normal(loc=mu_parallel,scale=sigma_parallel) # dimensionless
        self.velocity_vector[1] = np.random.normal(loc=mu_perp,scale=sigma_perp) # dimensionless
        self.velocity_vector[2] = np.random.normal(loc=mu_cross,scale=sigma_cross) # dimensionless
    
    def update_position_vector(self,L_i,mu_rnd,phi_rnd):
        """
        Update the position vector of the point.
        
        Parameters
        ----------
        L_i: float
            Comoving distance from last point, in Mpc.
        mu_rnd: float
            Random mu=cos(theta) with respect to the photon's trajectory.
        phi_rnd: float
            Random phi with respect to the photon's trajectory.
        
        """
        
        # Set a displacement vector in a frame aligned with the photon's 
        # direction from the previous iteration
        Delta_r_vector = np.array([L_i*mu_rnd,
                                   L_i*np.sqrt(1.-mu_rnd**2)*np.cos(phi_rnd),
                                   L_i*np.sqrt(1.-mu_rnd**2)*np.sin(phi_rnd)]
                                  ) # Mpc
        # Convert current photon's vector to spherical coordinates
        r_curr = norm(self.position_vector) # Mpc
        if r_curr == 0.:
            mu_curr = 1.
            phi_curr = 0.
        else:
            mu_curr = self.position_vector[0]/r_curr
            if mu_curr**2 == 1.:
                phi_curr = 0.
            else:
                phi_curr = np.arctan2(self.position_vector[2],self.position_vector[1])
        # Rotate displacement vector so it will be measured now from the "grid's" frame
        Delta_r_vector = calculate_rotation_matrix(mu_curr,phi_curr).dot(Delta_r_vector)  # Mpc
        # Update position vector
        self.position_vector += Delta_r_vector # Mpc
        # Sanity check: the norm of the position vector, i.e. the distance of
        #               the point from the source, can be also computed with
        #               the cosine theorem (see Eq. A4 in arXiv: 2101.01777)
        r_curr = np.sqrt(r_curr**2+L_i**2+2.*r_curr*L_i*mu_rnd) # Mpc
        if abs(1.-norm(self.position_vector)/r_curr) > 1.e-3:
            print(f"Warning at z={self.redshift}: position vector was not updated correctly")
    
    def compute_dtau_2_dL(self):
        """
        Compute dtau/dL, given redshift and apparent frequency.
        This is how optical depth tau changes with the comoving distance L.
        
        """
        # Compute cross section
        if self.sim_params.INCLUDE_TEMPERATURE:
            sigma_Lya = compute_Lya_cross_section(self.apparent_frequency,self.cosmo_params.T,self.sim_params.CROSS_SECTION) # m^2
        else:
            sigma_Lya = compute_Lya_cross_section(self.apparent_frequency,0.,self.sim_params.CROSS_SECTION) # m^2
        # Number density of neutral hydrogen
        # Note we assume homogeneity here
        n_HI = self.cosmo_params.n_H_z0*self.cosmo_params.x_HI*(1.+self.redshift)**3 # m^-3
        # This the integrand for the tau integral.
        # We divide by (1+z) as "L" here is comoving distance (not proper)
        dtau_2_dL = n_HI*sigma_Lya/(1.+self.redshift) # 1/m
        return dtau_2_dL


#%% Class for managing a single photon data

class PHOTON_POINTS_DATA():
    """
    Class for managing a single photon data.
    
    Parameters
    ----------
    z_abs_data: :class:`~COSMO_POINT_DATA`
        Data of the first point in the simulation (the absorption point).
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    sim_params: :class:`~SIM_PARAMS`
        The simulation parameters.
    random_seed: float
        The random seed to be used for this photon.
    
    The class stores all the scattering points of the photon (starting from
    the absorption point) in a list that can be accessed via `points_data`.
    """
    
    def __init__(self,
                 z_abs_data,
                 cosmo_params,
                 sim_params,
                 random_seed
                 
    ):
        self.z_abs = z_abs_data.redshift
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.points_data = [z_abs_data]
        self.random_seed = random_seed
    
    def append(self,point_data):
        """
        Append current point to this object.
        
        Parameters
        ----------
        point_data: :class:`~COSMO_POINT_DATA`
            New data point to append.
        """
        
        self.points_data.append(point_data)
        
    def draw_first_point(self):
        """
        Draw the first point for this photon.
        This is done from the analytical fit of Loeb & Rybicki 1999 
        (arXiv: astro-ph/9902180).
        
        Returns
        -------
        point_data: :class:`~COSMO_POINT_DATA`
            Data of the first point.
        """
        
        # Copy the content of z_abs_data into z_ini_data.
        z_abs_data = self.points_data[0]
        z_ini_data = z_abs_data.copy()
        # Find the next redshift from the initial frequency shift
        z_ini = (1.+self.z_abs)*(1.+ self.sim_params.Delta_nu_initial) - 1.
        z_ini_data.redshift = z_ini
        # Define initial frequency at z_ini
        z_ini_data.apparent_frequency = (1.+z_ini)/(1.+self.z_abs) # dimensionless (in units of Lya frequency)
        # Correct initial frequency due to temperature
        if self.sim_params.INCLUDE_TEMPERATURE:
            # Draw a random thermal velocity from Gaussian distribution
            # Note: here we do not divide the scale by sqrt(2) as we do in simulate_one_photon.
            #       This is because we are looking for the relative thermal velocity, so the variance is two times larger
            #       (since the thermal velocities are not correlated)
            Delta_nu = np.sqrt(2*k_B*self.cosmo_params.T/self.cosmo_params.m_b/c**2) # dimensionless (in units of nu_Lya)
            v_thermal_rel_parallel = np.random.normal(scale=Delta_nu)
            z_ini_data.apparent_frequency /= (1.-v_thermal_rel_parallel)
        if self.sim_params.INCLUDE_VELOCITIES:
            # Compute the smoothed 1D velocity RMS in z_ini
            z_ini_data.compute_RMS()
            # Draw velocity at z_prime based on the velocity vector at z_prime_old
            z_ini_data.draw_conditional_velocity_vector(z_abs_data)
            # Compute parallel component of relative velocity with respect to the last point
            # Remember: the first component in our velocity vector is always aligned with the photon's trajectory
            v_rel_parallel = z_ini_data.velocity_vector[0] - z_abs_data.velocity_vector[0] # dimensionless
            # Correct initial frequency due to peculiar velocity
            z_ini_data.apparent_frequency /= (1.-v_rel_parallel)
        # Draw the position vector from uncorrelated Gaussian distributions
        tilde_nu = np.abs(z_ini_data.apparent_frequency-1.)/self.cosmo_params.Delta_nu_star(self.z_abs) # dimensionless
        scale = np.sqrt(2./9.*tilde_nu**3)*self.cosmo_params.r_star(self.z_abs) # Mpc
        z_ini_data.position_vector = np.random.normal(scale=scale,size=3) # Mpc
        # Return output
        return z_ini_data
        
        
    def plot_photon_trajectory(self,
                               scale=5.,
                               ax = None,
                               **kwargs):
        """
        Plot the trajectory of this photon.
        
        Parameters
        ----------
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
        for i in range(len(self.points_data)):
            z_i_data = self.points_data[i]
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
    
    def plot_apparent_frequency(self,
                                ax = None,
                                **kwargs):
        """
        Plot the evolution of the apparent frequncy of this photon 
        in the gas frame.
        
        Parameters
        ----------
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
        for i in range(len(self.points_data)):
            z_i_data = self.points_data[i]
            z_list.append(z_i_data.redshift)
            nu_list.append(z_i_data.apparent_frequency)
        z_array = np.array(z_list)
        nu_array = np.array(nu_list)
        ax.loglog((z_array-self.z_abs)/(1.+self.z_abs),nu_array-1.,**kwargs)
        # Add a Lymann beta horizontal dashed line
        ax.axhline(y=Lyman_beta-1.,ls='--',color='k')
        # Add a diffusion horizontal dotted line
        ax.axhline(y=self.cosmo_params.Delta_nu_star(self.z_abs),ls=':',color='k')
        ax.axvline(x=self.cosmo_params.Delta_nu_star(self.z_abs),ls=':',color='k')
        ax.set_xlabel('$(z-z_\\mathrm{abs})/(1+z_\\mathrm{abs})$',fontsize=25)
        ax.set_ylabel('$\\nu/\\nu_\\alpha-1$',fontsize=25)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        if "label" in kwargs:
            ax.legend(fontsize=20)
        # Return output
        return fig, ax

    def plot_distance(self,
                      intermediate_pts = True,
                      ax = None,
                      **kwargs):
        """
        Plot the distance evolution of this photon with respect to the 
        absorption point.
        
        Parameters
        ----------
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
        for i in range(len(self.points_data)):
            z_i_data = self.points_data[i]
            z_list.append(z_i_data.redshift)
            r_list.append(norm(z_i_data.position_vector)) # Mpc
            if intermediate_pts:
                if i < len(self.points_data)-1:
                    z_i_plus1_data = self.points_data[i+1]
                    z_i_plus1 = z_i_plus1_data.redshift
                    z_i = z_i_data.redshift
                    # N is the number of grid points between z_i and z_{i+1}
                    N = int(self.cosmo_params.R_SL(z_i,z_i_plus1)/self.sim_params.Delta_L)
                    for n in np.arange(1,N):
                        z_prime = self.cosmo_params.R_SL_inverse(z_i,n*self.sim_params.Delta_L)
                        z_list.append(z_prime)
                        w = n/N
                        z_prime_position_vector = (1.-w)*z_i_data.position_vector # Mpc
                        z_prime_position_vector += w*z_i_plus1_data.position_vector # Mpc
                        r_list.append(norm(z_prime_position_vector)) # Mpc
        z_array = np.array(z_list)
        r_array = np.array(r_list)
        ax.loglog((z_array-self.z_abs)/(1.+self.z_abs),r_array,**kwargs)
        # Add a Lymann beta horizontal dashed line
        ax.axhline(y=self.cosmo_params.R_SL(self.z_abs,(1.+self.z_abs)*Lyman_beta-1.),ls='--',color='k')
        # Add a diffusion horizontal dotted line
        ax.axhline(y=self.cosmo_params.r_star(self.z_abs),ls=':',color='k')
        ax.axvline(x=self.cosmo_params.Delta_nu_star(self.z_abs),ls=':',color='k')
        ax.set_xlabel('$(z-z_\\mathrm{abs})/(1+z_\\mathrm{abs})$',fontsize=25)
        ax.set_ylabel('Distance [Mpc]',fontsize=25)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        if "label" in kwargs:
            ax.legend(fontsize=20)
        # Return output
        return fig, ax

#%% Class for managing data of all photons in the simulation

class ALL_PHOTONS_DATA():
    """
    Class for managing data of all photons in the simulation.
    
    Parameters
    ----------
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    sim_params: :class:`~SIM_PARAMS`
        The simulation parameters.
    
    The class stores all the photons of the simulation in a list that can be 
    accessed via `photons_data`.
    """
    
    def __init__(self,
                 cosmo_params,
                 sim_params
    ):
        self.z_abs = sim_params.z_abs
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.photons_data = []
        # Determine when to stop the simulation
        if not self.sim_params.x_stop is None: 
            r_stop = self.sim_params.x_stop*self.cosmo_params.r_star(self.z_abs) # Mpc
            z_stop = self.cosmo_params.R_SL_inverse(self.z_abs,r_stop) # final redshift
            self.nu_stop = (1.+z_stop)/(1.+self.z_abs) # frequency to stop the simulation (in units of Lya frequency)
            if self.nu_stop > Lyman_beta:
                self.nu_stop = Lyman_beta
        else:
            self.nu_stop = Lyman_beta
            z_stop = (1.+self.z_abs)*self.nu_stop - 1. # final redshift
            r_stop = self.cosmo_params.R_SL(self.z_abs,z_stop) # Mpc
            self.sim_params.x_stop = r_stop/self.cosmo_params.r_star(self.z_abs) # dimensionless
        # Compute the characteristic spectral width that is associated with the
        # IGM temperature.
        # Note that in our dimensionless units, Delta_nu also equals to the
        # rms of the thermal velocity, and we take advantage of this when we
        # drawn a random thermal velocity.
        if self.sim_params.INCLUDE_TEMPERATURE and (self.cosmo_params.T > 0.):
            self.Delta_nu = np.sqrt(2*k_B*self.cosmo_params.T/self.cosmo_params.m_b/c**2) # dimensionless (in units of nu_Lya)
            self.a = A_alpha_dimensionless/4/np.pi/self.Delta_nu
        else:
            self.Delta_nu = 0.
            self.a = np.inf
        # Make interpolation tables for the velocity rms and correlation 
        # coefficients
        if self.sim_params.INCLUDE_VELOCITIES and self.sim_params.USE_INTERPOLATION_TABLES:
            self.make_interpolation_tables()
        if self.sim_params.ANISOTROPIC_SCATTERING and not self.sim_params.STRAIGHT_LINE:
            self.make_mu_distribution_tables()
    
    def make_mu_distribution_tables(self):
        """
        Make interpolation tables for the inverse mu CDF, according to Eq. (20)
        in arXiv: 2311.03447.
        """
        
        self.mu_table_core = inverse_mu_CDF(11./24.,3./24.)
        self.mu_table_wing = inverse_mu_CDF(3./8.,3./8.)
    
    def make_interpolation_tables(self):
        """
        Make interpolation tables for the velocity rms and correlation 
        coefficients.
        """
        
        print("Now making interpolation tables...")
        # Create a redshift array that is identical to the redshift array
        # in the simulation (it only depends on Delta_L)
        z_end = (1.+self.z_abs)*self.nu_stop-1.
        if not self.sim_params.STRAIGHT_LINE:
            z_list = [self.z_abs, (1.+self.z_abs)*(1.+self.sim_params.Delta_nu_initial)-1.]
        else:
            z_list = [self.z_abs]
        while z_list[-1] < 1.02*z_end:
            z_list.append(self.cosmo_params.R_SL_inverse(z_list[-1],self.sim_params.Delta_L))
        z_array = np.array(z_list)
        # Create a velocity rms array for each redshift in z_array
        rms_array = np.zeros_like(z_array)
        for zi_ind, zi in enumerate(z_array):
            rms_array[zi_ind] = self.cosmo_params.compute_RMS(zi,self.sim_params.Delta_L)
        # Create correlation coefficients arrays for the velocities
        if not self.sim_params.NO_CORRELATIONS:
            rho_parallel_matrix = np.zeros((len(z_array),len(z_array)))
            rho_perp_matrix = np.zeros((len(z_array),len(z_array)))
            for zi_ind, zi in enumerate(z_array):
                for zj_ind, zj in enumerate(z_array):
                    # To save time, we only compute the upper elements of the matrix.
                    # No need to compute all elements because we are mostly interested
                    # in small scales correlations
                    if zj > zi and zj_ind < zi_ind + 10:
                        rho_parallel_matrix[zi_ind,zj_ind], rho_perp_matrix[zi_ind,zj_ind] = self.cosmo_params.compute_2_point_correlation(z1=zi,
                                                                                                                                        z2=zj,
                                                                                                                                        v1_1D_rms=rms_array[zi_ind],
                                                                                                                                        v2_1D_rms=rms_array[zj_ind])            
            # Symmetrize matrices and put 1 on the diagonal
            rho_parallel_matrix += rho_parallel_matrix.T + np.eye(len(z_array))
            rho_perp_matrix += rho_perp_matrix.T + np.eye(len(z_array))
        # Create interpolation tables
        self.cosmo_params.interpolate_RMS = interp1d(z_array, rms_array, kind='cubic')
        if not self.sim_params.NO_CORRELATIONS:
            self.cosmo_params.interpolate_rho_parallel = RectBivariateSpline(z_array, z_array, rho_parallel_matrix)
            self.cosmo_params.interpolate_rho_perp = RectBivariateSpline(z_array, z_array, rho_perp_matrix)
        # Save also z_array because sometimes the interpolation fails 
        # as the input is above the interpolation range
        self.cosmo_params.redshift_grid = z_array
    
    def append(self,photon_data):
        """
        Append current photon to this object.
        
        Parameters
        ----------
        point_data: :class:`~PHOTON_POINTS_DATA`
            New photon to append.
        """
    
        self.photons_data.append(photon_data)
    
    def plot_photon_trajectory(self,
                               photon_number,
                               scale=5.,
                               ax = None,
                               **kwargs):
        """
        Plot the trajectory of a photon.
        
        Parameters
        ----------
        photon_number: int
            The ID of the requested photon in the simulation.
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
        
        return self.photons_data[photon_number].plot_photon_trajectory(scale=scale,ax=ax,**kwargs)
    
    def plot_apparent_frequency(self,
                                photon_number,
                                ax = None,
                                **kwargs):
        """
        Plot the evolution of the apparent frequncy of a photon 
        in the gas frame.
        
        Parameters
        ----------
        photon_number: int
            The ID of the requested photon in the simulation.
        ax: Axes, optional
            The matplotlib Axes object on which to plot. Otherwise, created.
        kwargs:
            Optional keywords to pass to :func:`maplotlib.plot`.
        
        Returns
        -------
        fig, ax:
            figure and axis objects from matplotlib.
        """
        
        return self.photons_data[photon_number].plot_apparent_frequency(ax=ax,**kwargs)
    
    def plot_distance(self,
                      photon_number,
                      intermediate_pts = True,
                      ax = None,
                      **kwargs):
        """
        Plot the distance evolution of a photon with respect to the 
        absorption point.
        
        Parameters
        ----------
        photon_number: int
            The ID of the requested photon in the simulation.
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
        
        return self.photons_data[photon_number].plot_distance(ax=ax,intermediate_pts=intermediate_pts,**kwargs)
               
    def simulate_one_photon(self,random_seed):
        """
        Simulate one photon, given a random seed.
        
        Parameters
        ----------
        random_seed: int
            The random seed of this photon.
        
        This is the heart of the code. Given the random seed, a random velocity
        vector is drawn at z_abs, where the apparent frequency of the photon
        is initialized to Lyman alpha. The photon then propoagates backwards 
        in time and a random optical depth is drawn from exp(-tau) distribution. 
        During its trajectory, the frequency of the photon is blueshifted to 
        higher frequencies. Once the photon has traversed a distance that 
        corresponds to an optical depth that is greater than the threshold, 
        the photon scatters and gains a new random direction. The data of each 
        scattering point (which can be viewed as a potential source of emission) 
        is added to the object. This process continues until the apparent 
        photon's frequency crosses Lyman beta.
        
        """
        
        # Set random seed for this photon
        np.random.seed(random_seed)
        # Initialize a photon in z_abs with Lyman alpha frequency (1 
        # in our units, since we normalize frequency by nu_Lya)
        z_abs_data = COSMO_POINT_DATA(redshift=self.z_abs,
                                      cosmo_params=self.cosmo_params,
                                      sim_params=self.sim_params,
                                      apparent_frequency=1.)
        if self.sim_params.INCLUDE_VELOCITIES:
            # Compute the smoothed 1D velocity RMS in z_abs
            z_abs_data.compute_RMS()
            # Draw a random velocity at z_abs.
            # Velocities are dimensionless in this code as they are normalized
            # by c.
            z_abs_data.velocity_vector = np.random.normal(scale=z_abs_data.velocity_1D_rms,size=3) # dimensionless
        # Create a photon data object and initialize it with the point at z_abs
        photon_data = PHOTON_POINTS_DATA(z_abs_data=z_abs_data.copy(),
                                         cosmo_params=self.cosmo_params,
                                         sim_params=self.sim_params,
                                         random_seed=random_seed)
        # Draw the first position of the photon outside the origin.
        # This is the diffusion regime, where analytical result can be used.
        if not self.sim_params.STRAIGHT_LINE:
            z_ini_data = photon_data.draw_first_point()
            photon_data.append(z_ini_data.copy())
            # Initialize z_i to be at z_ini
            # Each z_i corresponds to a new scattering point
            z_i_data = z_ini_data.copy()
        else:
            z_i_data = z_abs_data.copy()
        # Compute tau integrand at z_ini
        dtau_2_dL_curr = z_i_data.compute_dtau_2_dL() # 1/m
        # Scatter the photon until we reached final frequency
        while z_i_data.apparent_frequency < self.nu_stop:
            # Draw random optical depth from an exponential distribution
            tau_rnd = -np.log(np.random.rand())
            # Initialize numerical integral for tau
            tau_integral = 0.
            dtau_2_dL_prev = dtau_2_dL_curr # 1/m
            # Initialize z' to be at z_i
            # z' is a dummy variable, used for the numerical integration of tau 
            z_prime_data = z_i_data.copy()
            # Calculate the optical depth numerically, until we crossed the
            # randomly drawn threshold value
            while tau_integral < tau_rnd and z_prime_data.apparent_frequency < self.nu_stop:
                # Update tau integrand
                dtau_2_dL_prev = dtau_2_dL_curr # 1/m
                # Set z_prime_old to be z_prime
                # We will use the velocity in z_prime_old to draw a correlated
                # velocity at the new z_prime
                z_prime_old_data = z_prime_data.copy()
                # Update z_prime
                next_redshift = self.cosmo_params.R_SL_inverse(z_prime_data.redshift,self.sim_params.Delta_L)
                z_prime_data = COSMO_POINT_DATA(redshift=next_redshift,
                                                cosmo_params=self.cosmo_params,
                                                sim_params=self.sim_params)
                # We don't need to track the photon's position at every z',
                # and instead we keep it to be in z_prime_old (which was set to
                # be at z_i)
                z_prime_data.position_vector = z_prime_old_data.position_vector
                if self.sim_params.INCLUDE_VELOCITIES:
                    # Update roatation matrix in z'
                    z_prime_data.rotation_matrix = z_prime_old_data.rotation_matrix
                    # Compute the smoothed 1D velocity RMS in z'
                    z_prime_data.compute_RMS()
                    # Draw velocity at z_prime based on the velocity vector at z_prime_old
                    z_prime_data.draw_conditional_velocity_vector(z_prime_old_data)
                    # Compute parallel component of relative velocity with respect to the last point
                    # Remember: the first component in our velocity vector is always aligned with the photon's trajectory
                    v_rel_parallel = z_prime_data.velocity_vector[0] - z_prime_old_data.velocity_vector[0] # dimensionless
                # Calculate apparent frequency at z_prime. 
                # First, we blueshift the apparent frequency from previous redshift
                z_prime_data.apparent_frequency = z_prime_old_data.apparent_frequency*(1.+z_prime_data.redshift)/(1.+z_prime_old_data.redshift) # dimensionless
                if self.sim_params.INCLUDE_VELOCITIES:
                    # Then, we Doppler shift it
                    # Sign was chosen such that nu_app is larger when v_rel_parallel > 0.
                    #   If v_rel_parallel > 0, the emitter (at z_prime) moves away 
                    #   from the absorber (at z_prime_old), which has a known a 
                    #   frequency. Thus, the emitter must have a larger frequency 
                    #   in order to compensate for the Doppler shift, which tends to lower
                    #   the frequency in the absorber's frame
                    z_prime_data.apparent_frequency /= (1.-v_rel_parallel) # dimensionless
                # Compute tau integrand at z' 
                dtau_2_dL_curr = z_prime_data.compute_dtau_2_dL() # 1/m
                # Compute integral of optical depth numerically 
                # (this is simple integration by the trapezoidal rule)
                dtau_2_dL = (dtau_2_dL_prev + dtau_2_dL_curr)/2. # 1/m
                dtau = dtau_2_dL * self.sim_params.Delta_L * Mpc_to_meter # dimensionless
                tau_integral += dtau  # dimensionless
            # Draw a random direction in which the photon has propagated
            if self.sim_params.STRAIGHT_LINE:
                mu_rnd = 1.
                phi_rnd = 0.
            else:
                if self.sim_params.ANISOTROPIC_SCATTERING:
                    # Draw random mu from the phase function,
                    # given by Eq. (20) in arXiv: 2311.03447
                    if abs(z_i_data.apparent_frequency-1.) < 0.2*self.Delta_nu:
                        mu_rnd = self.mu_table_core(np.array([np.random.rand()]))[0]
                    else:
                        mu_rnd = self.mu_table_wing(np.array([np.random.rand()]))[0]
                else:
                    # Draw random mu from a uniform distribution
                    mu_rnd = -1.+2.*np.random.rand()
                # phi is always drawn from a uniform distribution
                phi_rnd = 2.*np.pi*np.random.rand()
            # Compute comoving distance between z_i and z_{i+1} (assumed to be z_prime),
            # according to the straight line formula
            L_i = self.cosmo_params.R_SL(z_i_data.redshift,z_prime_data.redshift) # Mpc
            # Update scattering event (it is assumed that z_{i+1} = z_prime)
            z_i_data = z_prime_data.copy()
            # Update position vector of the photon
            z_i_data.update_position_vector(L_i,mu_rnd,phi_rnd)
            # Append z_{i+1} to the lists of scattering events
            photon_data.append(z_i_data.copy())
            # If we exited the tau loop because we exceeded tau_rnd,
            # then we have more upcoming scattering events!
            if tau_integral >= tau_rnd:
                # For the next scattering event, we need to rotate the velocity vector
                if self.sim_params.INCLUDE_VELOCITIES:
                    z_i_data.rotate(mu_rnd,phi_rnd)
                # Change frequency of scattered photon due to recoil
                if self.sim_params.INCLUDE_RECOIL and self.sim_params.INCLUDE_TEMPERATURE:
                    # Draw a random thermal velocity vector. The perpendicular component is drawn from a normal distribution
                    # while the parallel component is drawn from Eq. (25) in arXiv: 2311.03447.
                    # Note that in our dimensionless units, Delta_nu also equals the rms of thermal velocity
                    # Also note that we only need two components for the thermal velocity, not three
                    v_thermal_perp = np.random.normal(scale=self.Delta_nu/np.sqrt(2.)) # dimensionless
                    if self.cosmo_params.T > 0.:
                        v_thermal_parallel = self.Delta_nu*draw_from_voigt_distribution((z_i_data.apparent_frequency-1.)/self.Delta_nu,self.a) # dimensionless
                    else:
                        v_thermal_parallel = 0.
                    # Update frequency according to Eq. (23) in arXiv: 2311.03447
                    z_i_data.apparent_frequency /= 1. + (1.-mu_rnd)*h_P*z_i_data.apparent_frequency*nu_Lya/(m_H*c**2) # dimensionless
                    z_i_data.apparent_frequency *= 1. + (mu_rnd - 1.)*v_thermal_parallel + np.sqrt(1.-mu_rnd**2)*v_thermal_perp # dimensionless
                    # Also update tau integrand
                    dtau_2_dL_curr = z_i_data.compute_dtau_2_dL() # 1/m
            # Otherwise, we reached beyond final frequency and we can stop the simulation for this photon
            else:
                break
        # Append the data of this photon to the object
        self.append(photon_data)
        
    def simulate_N_photons(self):
        """
        Simulate many photons.
        
        Many different photons are simulated, each of which with a different
        random seed. The total number of simulated photons is determined by
        self.sim_params.N.
        """
        
        # Initalize random seed for the first photon
        random_seed = int(self.sim_params.z_abs)
        
        # Simulate N photons
        for n  in tqdm.tqdm(range(self.sim_params.N),
                            desc="",
                            unit="photons",
                            disable=False,
                            total= self.sim_params.N):
            self.simulate_one_photon(random_seed)
            # Use a different random seed for the next photon
            random_seed += 1

class SIM_DATA():
    """
    Class for collecting data from the simulation.
    
    Parameters
    ----------
    all_photons_data: :class:`~ALL_PHOTONS_DATA`
        Object that contains all data points of all photons in the simulation.
    quantity: string, optional
            Determines the quantity for which the data is collected. 
            There are two options:
                - 'distance': the distance from the absorption point, normalized
                              by the comoving straight-line distance between
                              z_abs and z_em.
                - 'velocity': the parallel component of the relative velocity
                              with respect to the absorption point, normalized
                              by c.
    intermediate_pts: bool, optional
            If this flag is True, then intermediate points (i.e. at z') are
            also taken into account. If false, then only the points in which 
            the photon has scattered, z_i, are taken into account.
    
    The class collects all the data in two lists that can be accessed via 'x_list' and 'y_list'.
    x_list is a list that contains data points of x_em (comoving distance between z_abs and z_em, 
    normalized by the comoving diffusion scale), while y_list contains the data points that correspond 
    to x_list. 
    """
    
    def __init__(self,
                 all_photons_data,
                 quantity='distance',
                 intermediate_pts = True
    ):
        self.z_abs = all_photons_data.sim_params.z_abs
        self.quantity = quantity
        self.intermediate_pts = intermediate_pts
        self.cosmo_params = all_photons_data.cosmo_params
        self.sim_params = all_photons_data.sim_params
        # Check input
        if not (quantity == 'distance' or quantity == 'velocity'):
            raise ValueError("'quantity' can only be 'distance' or 'velocity'.")
        if quantity == 'velocity' and not all_photons_data.sim_params.INCLUDE_VELOCITIES:
            raise ValueError('Requested to collect velocity data, but no velocity data is found.'
                             ' Change INCLUDE_VELOCITIES to True.')
        # Prepare lists
        z_em_list = []
        y_list = []
        # Gather data into lists
        for i in range(len(all_photons_data.photons_data)):
            photon_data = all_photons_data.photons_data[i]
            z_abs_data = photon_data.points_data[0]
            for j in range(len(photon_data.points_data)):
                z_i_data = photon_data.points_data[j]
                z_i = z_i_data.redshift
                if not z_i == self.sim_params.z_abs:
                    # Append z_i to z_em list
                    z_em_list.append(z_i)
                    # Compute distance from absorption point
                    r = norm(z_i_data.position_vector) # Mpc
                    # In case of distance, we normalize by R_SL
                    if quantity == 'distance':
                        y_list.append(r/self.cosmo_params.R_SL(self.sim_params.z_abs,z_i))
                    # In case of velocity, we need to rotate the velocity vector
                    # to the grid's frame, subtract the velocity vector at 
                    # z_abs, and dot product with a unit vector that connects
                    # the point to the source
                    elif quantity == 'velocity':
                        # Rotate velocity vector to the grid's fixed frame (as defined in z_abs)
                        # and compute the relative velocity between z_i and z_abs
                        v_rel_vector = (z_i_data.rotation_matrix.dot(z_i_data.velocity_vector)
                                        -z_abs_data.velocity_vector) # dimensionless
                        # Make a dot product with unit vector that points to the i'th position
                        # in order to extract the parallel component
                        v_rel_parallel = v_rel_vector.dot(z_i_data.position_vector)/r # dimensionless
                        # Append relative velocity to lists
                        y_list.append(v_rel_parallel)
                    # If we want to include intermediate points, we linearly interpolate 
                    # the position vector between z_i and z_{i+1} as we know that the photon
                    # has traveled in a straight line between two adjacent scattering events
                    # Note: we also linearly interpolate the velocity vector, but this is not precise! 
                    if intermediate_pts:
                        if j < len(photon_data.points_data)-1:
                            z_i_plus1_data = photon_data.points_data[j+1]
                            z_i_plus1 = z_i_plus1_data.redshift
                            # N is the number of grid points between z_i and z_{i+1}
                            N = int(self.cosmo_params.R_SL(z_i,z_i_plus1)/self.sim_params.Delta_L)
                            for n in np.arange(1,N):
                                z_prime = self.cosmo_params.R_SL_inverse(z_i,n*self.sim_params.Delta_L)
                                z_em_list.append(z_prime)
                                w = n/N
                                z_prime_position_vector = (1.-w)*z_i_data.position_vector # Mpc
                                z_prime_position_vector += w*z_i_plus1_data.position_vector # Mpc
                                r = norm(z_prime_position_vector) # Mpc
                                if quantity == 'distance':
                                    y_list.append(r/self.cosmo_params.R_SL(self.sim_params.z_abs,z_prime))
                                elif quantity == 'velocity':
                                    z_prime_velocity_vector = (1.-w)*z_i_data.rotation_matrix.dot(z_i_data.velocity_vector) # dimensionless
                                    z_prime_velocity_vector += w*z_i_plus1_data.rotation_matrix.dot(z_i_plus1_data.velocity_vector) # dimensionless
                                    v_rel_vector = z_prime_velocity_vector-z_abs_data.velocity_vector # dimensionless
                                    v_rel_parallel = v_rel_vector.dot(z_prime_position_vector)/r # dimensionless
                                    y_list.append(v_rel_parallel)
        # Normalize comoving distances from the origin by the diffusion distance to get x_em data points
        x_list = np.array([self.cosmo_params.R_SL(self.z_abs,z) for z in z_em_list])/self.cosmo_params.r_star(self.z_abs) # dimensionless
        # Get lists
        self.x_list = x_list
        self.y_list = y_list
    
    def plot_scatter_all_photons(self,
                                 x_axis='distance',
                                 ax = None,
                                 **kwargs):
        
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
            x_list = self.x_list
        else:
            x_list = np.array([self.cosmo_params.R_SL_inverse(self.z_abs,x_em*self.cosmo_params.r_star(self.z_abs)) for x_em in self.x_list])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure
        # Plot scatter of the data
        ax.scatter(x_list,self.y_list,**kwargs)
        # Prettify plot
        if self.quantity == 'distance':
            ax.set_ylabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
        elif self.quantity == 'velocity':
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
    
    def get_histogram(self,
                      x_res = 0.1,
                      N_bins = 21):

        """
        Get 2D histogram from data.
        
        Parameters
        ----------
        x_res: float, optional
            Resolution (width) of the x_em bins.
        N_bins: int, optioanl
            Number of bins in the 2D histogram in the quantity-axis (y-axis).
        
        Returns
        -------
        histogram_data: :class:`~HISTOGRAM_DATA`
            An object containing the information of the 2D histogram.
        """
        
        # Set x_edges
        # This will make bins of width x_res
        max_x = np.round(self.sim_params.x_stop,1)
        x_edges = np.linspace(x_res/2.,max_x+x_res/2.,int(np.round(max_x/x_res))+1)
        # Set y_edges
        if self.quantity == 'distance':
            # I want that the first and last bins will be 0 and 1, with even spaces between the N_bins
            y_edges = np.linspace(0.,1.+1./(N_bins-1.),N_bins+1)-1./(2.*(N_bins-1.))
        else:
            y_edges = np.linspace(min(self.y_list),max(self.y_list),N_bins+1)
        # Get 2D histogram of the data
        H_matrix = np.histogram2d(self.x_list, self.y_list,[x_edges,y_edges])[0]
        x_bins = (x_edges[1:]+x_edges[:-1])/2.
        y_bins = (y_edges[1:]+y_edges[:-1])/2.
        # Return output
        return HISTOGRAM_DATA(cosmo_params=self.cosmo_params,
                              sim_params=self.sim_params,
                              H_matrix=H_matrix,
                              x_bins=x_bins,
                              y_bins=y_bins,
                              quantity=self.quantity)
    
    def find_fitting_params(self,
                            x_em,
                            x_res = 0.1):
        """
        Fit the data at a given x_em to an analytical function.
        If quantitiy = 'distance', the data is fitted to 
        a beta function.
        Otherwise, if quantitiy = 'velocity', the data is fitted to a Gaussian.
        
        Parameters
        ----------
        x_em: float
            Comoving straight-line distance between z_abs and z_em, normalized 
            by the diffusion scale at z_abs.
        x_res: float, optional
            Resolution (width) of the x_em bins.
        
        Returns
        -------
        dist_params: tuple
            The parameters of the fitted distibution. Depends on the quantity 
            of the data.
            There are two options:
                - 'distance': (alpha, beta) the parameters of the beta distribution.
                - 'velocity': (mu, sigma) the parameters of the Gaussian distribution.
        data_lims: tuple
            The minimum and maximum limits of the data at the input x_em.
        """

        # Set x_edges
        # This will make bins of width x_res
        max_x = np.round(self.sim_params.x_stop,1)
        x_edges = np.linspace(x_res/2.,max_x+x_res/2.,int(np.round(max_x/x_res))+1)
        # Collect only data that falls in the appropriate x_em bin
        bin_number = np.digitize(x_em,x_edges)
        data = np.array(self.y_list)[np.digitize(self.x_list,x_edges) == bin_number]
        data_lims = (data.min(), data.max())
        # Find mean and variance
        mean = data.mean()
        var = data.var(ddof=1)
        # Find fitting parameters
        if self.quantity == 'distance':
            # Fit to a beta distribution
            alpha = mean * ( (mean*(1.-mean))/var - 1.)
            beta = (1. - mean) * ( (mean*(1.-mean))/var - 1.)
            dist_params = (alpha, beta)
        elif self.quantity == 'velocity':
            dist_params = (mean, np.sqrt(var))
        # Return output
        return dist_params, data_lims
    
    def plot_fit(self,
                 x_em,
                 x_res = 0.1,
                 ax = None,
                 **kwargs):
        
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
        
        dist_params, data_lims = self.find_fitting_params(x_em=x_em,x_res=x_res)
        if self.quantity == 'distance':
            (alpha, beta) = dist_params
            x_array = np.linspace(0.,1.,1000)
        elif self.quantity == 'velocity':
            (mu, sigma) = dist_params
            x_array = np.linspace(data_lims[0],data_lims[1],1000)
        # Prepare figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure
        # Plot the analytical fit
        if self.quantity == 'distance':
            ax.plot(x_array, beta_dist.pdf(x_array,alpha,beta),**kwargs)
            # Prettify plot
            ax.set_ylabel('Radial distribution',fontsize=25)
            ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
        elif self.quantity == 'velocity':
            ax.plot(x_array, norm_dist.pdf(x_array,mu,sigma), **kwargs)
            ax.set_ylabel('Velocity distribution',fontsize=25)
            ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        if "label" in kwargs:
            ax.legend(fontsize=20)
        # Return output
        return fig, ax

#%% Class for containing histogram data

class HISTOGRAM_DATA():
    """
    Class for containing 2D histogram data.
    
    Parameters
    ----------
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    sim_params: :class:`~SIM_PARAMS`
        The simulation parameters.
    H_matrix: numpy array (Nx_bins X Ny_bins)
        The 2D histogram values.
    x_bins: numpy array (Nx_bins X 1)
        The bins values along the x-axis of the histogram.
    y_bins: numpy array (Ny_bins X 1)
        The bins values along the y-axis of the histogram.
    quantity: string
        Determines the quantity for which the histogram will be achieved. 
        There are two options:
            - 'distance': the distance from the absorption point, normalized
                          by the comoving straight-line distance between
                          z_abs and z_em.
            - 'velocity': the parallel component of the relative velocity
                          with respect to the absorption point, normalized
                          by c.
    """
    
    
    def __init__(self,
                 cosmo_params,
                 sim_params,
                 H_matrix,
                 x_bins,
                 y_bins,
                 quantity):
        self.z_abs = sim_params.z_abs
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.H_matrix = H_matrix
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.quantity = quantity
        
    def plot_histogram(self,
                       z_em=None,
                       x_em=None,
                       ax = None,
                       **kwargs):
        
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
            x_em = self.cosmo_params.R_SL(self.z_abs,z_em)/self.cosmo_params.r_star(self.z_abs)
        if x_em > np.max(self.x_bins):
            raise ValueError(f"Requested input corresponds to x_em={x_em}, but maximum x_em in histogram data is {np.max(self.x_bins)}")
        elif x_em < np.min(self.x_bins):
            raise ValueError(f"Requested input corresponds to x_em={x_em}, but minimum x_em in histogram data is {np.min(self.x_bins)}")
        else:
            x_em_arg = np.argmin(np.abs(self.x_bins-x_em))
        # Extract 1D distribution
        f_array = self.H_matrix[x_em_arg,:] # dimensionless
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
        dr = np.diff(self.y_bins)[0]
        edges = np.concatenate((np.array([self.y_bins[0]-dr/2.]),
                                (self.y_bins[1:]+self.y_bins[:-1])/2.,
                                np.array([self.y_bins[-1]+dr/2.])))
        # Plot 1D histogram of the data, normalized to 1
        if self.quantity == 'distance':
            r_array = self.y_bins # dimensionless
            # Ensure that the bin at 0 and 1 are empty
            f_array[1] += f_array[0]
            f_array[0] = 0.
            if not self.sim_params.STRAIGHT_LINE:
                f_array[-2] += f_array[-1]
                f_array[-1] = 0.
            # Normalize
            f_array /= intg.simpson(f_array, x=r_array) # dimensionless
            # Plot!
            ax.bar(edges[:-1],f_array,width=np.diff(edges),**kwargs)
            # Prettify plot
            ax.set_ylabel('Radial distribution',fontsize=25)
            ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
        elif self.quantity == 'velocity':
            v_array = self.y_bins # dimensionless
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
    
    def fit_histogram(self,
                      z_em=None,
                      x_em=None):
        
        """
        Fit the 1D histogram at z_em to an analytical function.
        If the 1D histogram contains the information of the comoving
        distance from the absorption point, the histogram is fitted to 
        a beta function.
        Otherwise, if the 1D histogram contains the information of the 
        parallel component of the relative velocity with respect to the
        absorption point, the histogram is fitted to a Gaussian.
        
        Parameters
        ----------
        z_em: float, optional
            The redshift of the emitter. Cannot be passed if x_em is 
            also passed.
        x_em: float, optional
            Comoving straight-line distance between z_abs and z_em, normalized 
            by the diffusion scale at z_abs. Cannot be passed if z_em is 
            also passed.
        
        Returns
        -------
        f_array: numpy array
            The normalized distribution.
        x_array: numpy array
            Depends on the quantity that is stored in the histogram data.
            There are two options:
                - 'distance': the distance from the absorption point, normalized
                              by the comoving straight-line distance between
                              z_abs and z_em.
                - 'velocity': the parallel component of the relative velocity
                              with respect to the absorption point, normalized
                              by c.
        dist_params: tuple
            The parameters of the fitted distibution. Depends on the quantity 
            that is stored in the histogram data.
            There are two options:
                - 'distance': (alpha, beta) the parameters of the beta distribution.
                - 'velocity': (mu, sigma) the parameters of the Gaussian distribution.
        """
        
        # Check input
        if x_em == None and z_em == None:
            raise ValueError("Either 'x_em' or 'z_em' must be specified.")
        if (not x_em == None) and (not z_em == None):
            raise ValueError("'x_em' and 'z_em' cannot be both specified. Choose one of them.")
        # Find 1D data that corresponds to input
        if not z_em == None:
            x_em = self.cosmo_params.R_SL(self.z_abs,z_em)/self.cosmo_params.r_star(self.z_abs)
        if x_em > np.max(self.x_bins):
            raise ValueError(f"Requested input corresponds to x_em={x_em}, but maximum x_em in histogram data is {np.max(self.x_bins)}")
        elif x_em < np.min(self.x_bins):
            raise ValueError(f"Requested input corresponds to x_em={x_em}, but minimum x_em in histogram data is {np.min(self.x_bins)}")
        else:
            x_em_arg = np.argmin(np.abs(self.x_bins-x_em))
        # Extract 1D distribution
        f_array = self.H_matrix[x_em_arg,:] # dimensionless
        if self.quantity == 'distance':
            # Normalize distribution
            r_array = self.y_bins # dimensionless
            # Ensure that the bin at 0 and 1 are empty
            f_array[1] += f_array[0]
            f_array[0] = 0.
            if not self.sim_params.STRAIGHT_LINE:
                f_array[-2] += f_array[-1]
                f_array[-1] = 0.
            # Normalize
            f_array /= intg.simpson(f_array, x=r_array) # dimensionless
            # Guess parameters for initialization.
            alpha_initial = 1.+pow(x_em,1/2)
            beta_initial = 1.+pow(x_em,-1/2)
            initial_guess = (alpha_initial,beta_initial)
            bounds = ([1., 1.],[np.inf, np.inf])
            # Fit to a beta distribution
            alpha, beta, = curve_fit(beta_dist.pdf,r_array,f_array,initial_guess,bounds=bounds)[0]
            dist_params = (alpha, beta)
        elif self.quantity == 'velocity':
            v_array = self.y_bins # dimensionless
            # Normalize
            f_array /= intg.simpson(f_array, x=v_array) # dimensionless
            # Guess parameters for initialization.
            # We guess the Gaussian parameters from linear theory
            # Note: in theory, the standard deviation of the relative velocity
            #       should be sqrt[<v_abs^2> + <v_em^2> - 2*rho*<v_abs^2>^{1/2}*<v_em^2>^{1/2}].
            #       Because rho that is computed in compute_2_point_correlation
            #       smoothes the velocity field at a radius that corresponds to
            #       the distance between the points at z_abs and z_em, we 
            #       naively guess that the standard deviation is just <v_abs^2>^{1/2}
            #       (that should be fixed in the future...)
            mu_initial = 0.
            sigma_initial = self.cosmo_params.compute_RMS(self.z_abs, self.sim_params.Delta_L)
            # Fit to normal distribution
            mu, sigma = curve_fit(norm_dist.pdf,v_array, f_array,(mu_initial,sigma_initial))[0]
            dist_params = (mu, sigma)
        # Return output
        return f_array, self.y_bins, dist_params
    
    def plot_fit(self,
                 z_em=None,
                 x_em=None,
                 ax = None,
                 **kwargs):
        
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
            if self.quantity == 'distance':
                f_array, x_array, (alpha, beta) = self.fit_histogram(z_em=z_em)
            else:
                f_array, x_array, (mu, sigma) = self.fit_histogram(z_em=z_em)
        else:
            if self.quantity == 'distance':
                f_array, x_array, (alpha, beta) = self.fit_histogram(x_em=x_em)
            else:
                f_array, x_array, (mu, sigma) = self.fit_histogram(x_em=x_em)
        # Get more samples to draw nice functions
        x_array_new = np.linspace(x_array[0],x_array[-1],1000)
        # Prepare figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure
        # Plot the analytical fit
        if self.quantity == 'distance':
            ax.plot(x_array_new, beta_dist.pdf(x_array_new,alpha,beta),**kwargs)
            # Prettify plot
            ax.set_ylabel('Radial distribution',fontsize=25)
            ax.set_xlabel('$r/R_\\mathrm{SL}(z_\\mathrm{abs},z_\\mathrm{em})$',fontsize=25)
        elif self.quantity == 'velocity':
            ax.plot(x_array_new, norm_dist.pdf(x_array_new,mu,sigma), **kwargs)
            ax.set_ylabel('Velocity distribution',fontsize=25)
            ax.set_xlabel('$v_\\mathrm{rel}^{||}/c$',fontsize=25)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        if "label" in kwargs:
            ax.legend(fontsize=20)
        # Return output
        return fig, ax
        
#%% Set your parameters here
if __name__ == "__main__":
    # Set cosmological parameters
    cosmo_params = {}
    cosmo_params["h"] = 0.6736 # Hubble constant (in 100 km/sec/Mpc)
    cosmo_params["Omega_m"] = 0.3153 # Current matter density parameter
    cosmo_params["Omega_b"] = 0.0493 # Current baryon density parameter
    cosmo_params["A_s"] = 2.1e-9 # Amplitude of primordial curvator fluctuations in the pivot scale 0.05 Mpc^-1
    cosmo_params["n_s"] = 0.9649 # Spectral tilt of primordial curvator fluctuations
    cosmo_params["T"] = 1.e4 # Temperature of the IGM in the simulation, in Kelvin
    cosmo_params["x_HI"] = 1. # Fraction of neutral hydrogen in the simulation
    # Create a COSMO_PARAMS object
    cosmo_params = COSMO_PARAMS(cosmo_params)
    # Set simulation parameters
    sim_params = {}
    sim_params["z_abs"] = 10. # Redshift where the photon was absorbed
    #sim_params["x_stop"] = 1. # Distance from the source to stop the simulation, in units of the diffusion scale
    sim_params["N"] = int(5) # Number of photons to be simulated
    sim_params["Delta_L"] = 0.2 # "Grid" resolution, in Mpc
    sim_params["Delta_nu_initial"] = 2.e-4 # Initial frequency difference from Lyman alpha, in units of Lyman alpha frequency
    sim_params["INCLUDE_VELOCITIES"] = True # Do we want peculiar velocities?
    sim_params["NO_CORRELATIONS"] = False # Do we want to use the correlations in the velocity field?
    sim_params["USE_INTERPOLATION_TABLES"] = True # Do we want to use interpolation tables for velocity?
    sim_params["INCLUDE_TEMPERATURE"] = True # Do we want to include finite temperature
    sim_params["ANISOTROPIC_SCATTERING"] = True # Do we want to draw mu from a unifrom distribution?
    sim_params["INCLUDE_RECOIL"] = True # Do we want to account for atom recoil during scattering?
    sim_params["STRAIGHT_LINE"] = False # Do we want to assume straight-line approximation?
    sim_params["CROSS_SECTION"] = 'LORENTZIAN' # Type of cross-section. Can be LORENTZIAN or PEEBLES.
    # Create a SIM_PARAMS object
    sim_params = SIM_PARAMS(sim_params)
    #%% This is where the calculation begins
    # Run CLASS
    cosmo_params.run_CLASS()
    # Initialize output object
    all_photons_data = ALL_PHOTONS_DATA(cosmo_params,sim_params)
    # Simulate N photons
    all_photons_data.simulate_N_photons()
    # Calculate histogram
    if sim_params.N > 100:
        histogram_data = all_photons_data.get_histogram(N_bins=21)
        # Plot histogram at x_em = 1
        fig, ax = histogram_data.plot_histogram(x_em=1.,
                                                label='Histogram (simulation)')
        # Plot fit at x_em = 1
        fig, ax = histogram_data.plot_fit(x_em=1.,
                                          label='Fit (beta distribution)',
                                          color='r',
                                          lw=2,
                                          ax=ax)