"Module for Lyman alpha calculations."

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from .constants import *

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
