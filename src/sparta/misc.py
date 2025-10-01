"Module for miscellaneous functions of SPaRTA."

import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as intg
from scipy.special import voigt_profile

#%% Define some global parameters
c = 2.99792458e8 # Speed of light in m/sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
lambda_alpha = c/nu_Lya # Lya wavelength in m
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya

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

def compute_correlation_function(
    CLASS_OUTPUT,
    cosmo_params,
    r = 0.,
    r_smooth = 0.,
    z1 = 0.,
    z2 = 0.,
    kind1 = "density_m",
    kind2 = "density_m",
    normalization = True
    ):
    
    """
    Compute the 2-point correlation function of a field of type kind1 at redshift z1
    with a field of type kind2 at redshift z2. The fields can be smoothed by a top-hat
    filter in real space with radius r_smooth.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    r: float, optional
        The comoving distance in Mpc. Default is 0.
    r_smooth: float, optional
        The smoothing radius in Mpc.
    z1: float, optional
        The redshift of field of type kind1. Default is 0.
    z2: float, optional
        The redshift of field of type kind2. Default is 0.    
    kind1: str
        The kind of the first field: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    kind2: str
        The kind of the second field: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    normalization: bool
        Whether to normalize the correlation function with the RMS.
    
    Returns
    -------
    float:
        The 2-point correlation function.
    """
    
    transfer1, k_array = get_transfer_function(
        CLASS_OUTPUT = CLASS_OUTPUT,
        cosmo_params = cosmo_params,
        kind = kind1,
        z = z1,
        r = r
    )
    transfer2, k_array = get_transfer_function(
        CLASS_OUTPUT = CLASS_OUTPUT,
        cosmo_params = cosmo_params,
        kind = kind2,
        z = z2,
        r = r
    )
    # Power spectrum of primordial curvature fluctuations
    Delta_R_sq = cosmo_params.A_s*pow(k_array/0.05,cosmo_params.n_s-1.)
    # Window functions
    with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
        W_k_array = window_function_for_correlation(k_array*r,kind1,kind2)
        W_k_top_hat = top_hat_window_function(k_array*r_smooth)
    # Smooth transfer functions
    transfer1 *= W_k_top_hat
    transfer2 *= W_k_top_hat
    # Integrate to get the correlation function xi(r)
    integrand = Delta_R_sq * transfer1 * transfer2 * W_k_array /k_array # Mpc
    xi = intg.simpson(integrand, x=k_array) # dimensionless
    # Normalize if needed
    if normalization:
        xi /= np.sqrt(intg.simpson(Delta_R_sq * transfer1 * transfer1 / k_array, x=k_array)) # dimensionless
        xi /= np.sqrt(intg.simpson(Delta_R_sq * transfer2 * transfer2 / k_array, x=k_array)) # dimensionless
    # Return output
    return xi

def get_transfer_function(
    CLASS_OUTPUT,
    cosmo_params,
    kind,
    z = 0,
    r = 0
):
    """
    Get the transfer function from the output of CLASS.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    kind: str
        Kind of transfer function to get: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    z: float, optional
        The redshift of the transfer function. Default is 0.
    r: float, optional
        The comoving distance in Mpc (used for setting up k_array). Default is 0.
    
    Returns
    -------
    transfer: np.ndarray
        The transfer function.
    k_array: np.ndarray
        An array of the wavenumbers associated with the transfer function.
    """

    # Extract transfer function at redshift z
    k_CLASS = CLASS_OUTPUT.get_transfer(z=z)["k (h/Mpc)"]*cosmo_params.h # 1/Mpc
    Transfer_z = CLASS_OUTPUT.get_transfer(z=z)
    if kind == "density_m":
        transfer_CLASS = Transfer_z["d_m"]
    elif kind == "density_b":
        transfer_CLASS = Transfer_z["d_b"]
    elif kind == "v_parallel" or  kind == "v_perp":
        transfer_CLASS = Transfer_z["t_b"]/k_CLASS/np.sqrt(3.) # dimensionless
    else:
        raise ValueError("'kind' can only be 'density_m', 'density_b', 'v_parallel'  or 'v_perp'.")

    # Interpolate transfer function
    if r > 0:
        k_array = np.logspace(-4.,4.,10000)/r
        k_array = np.concatenate((np.linspace(min(k_CLASS),min(k_array),100),k_array))
        k_array = np.unique(k_array)
        k_array = np.sort(k_array)
    else:
        k_array = k_CLASS
    # TODO: Normally the phase of the transfer function is not interesting because we usually compute auto-covariance.
    #       It becomes relevant only when doing cross-covariance (e.g. delta with v_parallel)
    #       It is a bit tricky to do logarithmic interpolation while keeping the phase information.
    #       There's got to be a better way than what I'm doing here.
    sign = float(2 * np.all(transfer_CLASS > 0) - 1)
    transfer = sign * np.exp(
        interp1d(
            np.log(k_CLASS), 
            np.log(np.abs(transfer_CLASS)),
            kind='cubic',
            bounds_error=False,
            fill_value="extrapolate"
        )(np.log(k_array))
    )
    return transfer, k_array

def window_function_for_correlation(
    kr,
    kind1,
    kind2,
):
    if (
        (kind1 == "density_m" or kind1 == "density_b") 
        and 
        (kind2 == "density_m" or kind2 == "density_b")
        ):
        return window_function_density_density(kr)
    elif ((kind1 == "v_parallel") and (kind2 == "v_parallel")):
        return window_function_v_parallel_v_parallel(kr)
    elif ((kind1 == "v_perp") and (kind2 == "v_perp")):
        return window_function_v_perp_v_perp(kr)
    elif (
        (
            (kind1 == "density_m" or kind1 == "density_b") 
            and 
            (kind2 == "v_parallel")
        )
        or
            (kind2 == "density_m" or kind2 == "density_b") 
            and 
            (kind1 == "v_parallel")
            ):
        return window_function_density_v_parallel(kr)
    else:
        return 0.

def window_function_density_density(kr):
    return np.where(
        kr < 1.e-3,
        1.-(kr**2)/6.,
        np.sin(kr)/kr
    )
    
def window_function_v_parallel_v_parallel(kr):
    return np.where(
        kr < 1.e-3,
        1.-3.*(kr**2)/10.,
        (3.*(kr**2-2.)*np.sin(kr)+6.*kr*np.cos(kr))/kr**3
    )
    
def window_function_v_perp_v_perp(kr):
    return np.where(
        kr < 1.e-3,
        1.-(kr**2)/10.,
        3.*(np.sin(kr)-kr*np.cos(kr))/kr**3
    )

def window_function_density_v_parallel(kr):
    return np.where(
        kr < 1.e-3,
        kr/np.sqrt(3.),
        np.sqrt(3.)*(np.sin(kr)-kr*np.cos(kr))/kr**2
    )

def top_hat_window_function(kr):
    return np.where(
        kr < 1.e-3,
        1.-(kr**2)/10.,
        3.*(np.sin(kr)-kr*np.cos(kr))/kr**3
    )