"Module for computing cosmological quantities."

import numpy as np
from scipy.special import voigt_profile

#%% Define some global parameters
c = 2.99792458e8 # Speed of light in m/sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
lambda_alpha = c/nu_Lya # Lya wavelength in m
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya

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