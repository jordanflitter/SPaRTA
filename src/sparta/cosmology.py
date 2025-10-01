"Module for computing cosmological quantities."

import numpy as np
import scipy.integrate as intg
from scipy.interpolate import interp1d
from scipy.special import voigt_profile

#%% Define some global parameters
c = 2.99792458e8 # Speed of light in m/sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
lambda_alpha = c/nu_Lya # Lya wavelength in m
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya

def compute_Lya_cross_section(
    nu_apparent,
    T,
    flag='LORENTZIAN'
    ):

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

def compute_RMS(
    cosmo_params,
    CLASS_OUTPUT,
    z,
    r
    ):

    """
    Compute 1D velocity rms at redshift z, smoothed on a scale r.
    
    Parameters
    ----------
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
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
    k_CLASS = CLASS_OUTPUT.get_transfer(z=z)['k (h/Mpc)'][:]*cosmo_params.h # 1/Mpc
    Transfer_z = CLASS_OUTPUT.get_transfer(z=z)
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
    Delta_R_sq = cosmo_params.A_s*pow(k_arr/0.05,cosmo_params.n_s-1.) # dimensionless
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

def compute_Pearson_coefficient(
    cosmo_params,
    CLASS_OUTPUT,
    z1,
    z2,
    v1_1D_rms,
    v2_1D_rms,
    r=None
    ):
    
    """
    Compute velocity correlation coefficients for parallel 
    and perpendicular components, between two points at redshifts 
    z1 and z2.
    
    Parameters
    ----------
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
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
    k_CLASS = CLASS_OUTPUT.get_transfer(z=z1)['k (h/Mpc)'][:]*cosmo_params.h # 1/Mpc
    Transfer_z1 = CLASS_OUTPUT.get_transfer(z=z1)
    Transfer_z2 = CLASS_OUTPUT.get_transfer(z=z2)
    theta_b_z1_CLASS = Transfer_z1['t_b'][:] # 1/Mpc
    theta_b_z2_CLASS = Transfer_z2['t_b'][:] # 1/Mpc
    # Comoving distance between z1 and z2
    if r is None:
        r = cosmo_params.R_SL(z1,z2) # Mpc
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
    Delta_R_sq = cosmo_params.A_s*pow(k_arr/0.05,cosmo_params.n_s-1.)
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