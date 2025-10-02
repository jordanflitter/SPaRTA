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
    CLASS_OUTPUT,
    z,
    r_smooth,
    kind = "density_m"
    ):

    """
    Compute the RMS of a field at redshift z, smoothed on a scale r_smooth.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    z: float
        Redshift.
    r_smooth: float
        Smoothing radius, in Mpc.
    kind: str, optional
        The kind of field for the RMS is computed: options are "density_m", "density_b"
        "v_parallel" and "v_perp" (the RMS of the last two is the same,
        hence "velocity" is also accepted). Default is "density_m".
    
    Returns
    -------
    float:
        The RMS of the smoothed field.
    """

    if kind == "velocity":
        kind = "v_perp"

    variance = compute_correlation_function(
        CLASS_OUTPUT = CLASS_OUTPUT,
        z1 = z,
        z2 = z,
        r = 0.,
        r_smooth = r_smooth,
        kind1 = kind,
        kind2 = kind,
    )
    return np.sqrt(variance) # dimensionless

def compute_Pearson_coefficient(
    CLASS_OUTPUT,
    z1,
    z2,
    r,
    r_smooth,
    kinds_list,
    ):
    
    """
    Compute the Pearson correlation coefficient for multiple fields at redshifts 
    z1 and z2.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    z1: float
        Initial redshift.
    z2: float
        Final redshift.
    r: float
        Comoving distance between the two points, in Mpc.
    r_smooth: float
        Smoothing radius, in Mpc.
    kinds_list: list of tuples
        A list of the form [(kind1,kind2),(kind3,kind4),...]
        Each kind is a string specifying the type kind of the field for which the 
        Pearson coefficient is evaluated: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    
    Returns
    -------
    dict:
        Dictionary of the Pearson coeffcients. The keys of the dictionary are the strings in kinds_list,
        e.g. "kind1,kind2", "kind3,kind4", etc.
    """

    rho_dict = {}

    for kinds in kinds_list:
        kind1, kind2 = kinds
    
        rho = compute_correlation_function(
            CLASS_OUTPUT = CLASS_OUTPUT,
            z1 = z1,
            z2 = z2,
            r = r,
            r_smooth = r_smooth,
            kind1 = kind1,
            kind2 = kind2,
            normalization = True
        )
    
        # Sanity check: -1 <= rho <= 1
        if rho**2 > 1:
            print(f"Warning: At (z1,z2)={z1,z2} the Pearson correlation coefficient for kinds = ({kind1},{kind2}) is rho={rho}")
        
        rho_dict[f"{kind1},{kind2}"] = rho
    # Return output
    return rho_dict

def compute_correlation_function(
    CLASS_OUTPUT,
    r = 0.,
    r_smooth = 0.,
    z1 = 0.,
    z2 = 0.,
    kind1 = "density_m",
    kind2 = "density_m",
    normalization = False
    ):
    
    """
    Compute the 2-point correlation function of a field of type kind1 at redshift z1
    with a field of type kind2 at redshift z2. The fields can be smoothed by a top-hat
    filter in real space with radius r_smooth.
    
    Parameters
    ----------
    CLASS_OUTPUT: :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
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
    
    # Get transfer funcions
    transfer1, k_array = get_transfer_function(
        CLASS_OUTPUT = CLASS_OUTPUT,
        kind = kind1,
        z = z1,
        r = r
    )
    if z2 == z1 and kind2 == kind1:
        transfer2 = transfer1.copy()
    else:    
        transfer2, k_array = get_transfer_function(
            CLASS_OUTPUT = CLASS_OUTPUT,
            kind = kind2,
            z = z2,
            r = r
        )
    # Power spectrum of primordial curvature fluctuations
    A_s = CLASS_OUTPUT.get_current_derived_parameters(["A_s"])["A_s"]
    n_s = CLASS_OUTPUT.n_s()
    Delta_R_sq = A_s*pow(k_array/0.05,n_s-1.)
    # Window functions
    with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
        W_k_array = window_function_for_correlation(k_array*r,kind1,kind2)
        W_k_top_hat = top_hat_window_function(k_array*r_smooth)
    # Smooth transfer functions
    transfer1 *= W_k_top_hat
    transfer2 *= W_k_top_hat
    # Integrate to get the correlation function xi(r)
    integrand = Delta_R_sq * transfer1 * transfer2 * W_k_array /k_array # Mpc
    correlation = intg.simpson(integrand, x=k_array) # dimensionless
    # Normalize if needed
    if normalization:
        correlation /= np.sqrt(intg.simpson(Delta_R_sq * transfer1 * transfer1 / k_array, x=k_array)) # dimensionless
        correlation /= np.sqrt(intg.simpson(Delta_R_sq * transfer2 * transfer2 / k_array, x=k_array)) # dimensionless
    # Return output
    return correlation

def get_transfer_function(
    CLASS_OUTPUT,
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
    k_CLASS = CLASS_OUTPUT.get_transfer(z=z)["k (h/Mpc)"]*CLASS_OUTPUT.h() # 1/Mpc
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
        k_array = np.logspace(-4.,4.,10000)/r # 1/Mpc
        k_array = np.concatenate((np.linspace(min(k_CLASS),min(k_array),100),k_array)) # 1/Mpc
        k_array = np.unique(k_array) # 1/Mpc
        k_array = np.sort(k_array) # 1/Mpc
    else:
        k_array = k_CLASS
        # This array follows the same spacing transitions in the wavenumbers
        # listed in Transfers_z0.dat in 21cmFAST, but I added more samples in order to compute
        # more precisely the variance
        k_array = np.concatenate(
                (
                    np.logspace(-5.15, -1.49, 50),
                    np.logspace(-1.45, -0.258, 80),
                    np.logspace(-0.2083, 3.049, 100),
                )
        )
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
    """
    Compute the window function for the correlation function between two fields.
    
    Parameters
    ----------
    kr: float or np.ndarray
        The argument of the window function.
    kind1: str
        The kind of the first field: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    kind2: str
        The kind of the second field: options are "density_m", "density_b"
        "v_parallel" and "v_perp".
    
    Returns
    -------
    float or np.ndarray
        The desired window function.
    """

    if (
        (kind1 == "density_m" or kind1 == "density_b") 
        and 
        (kind2 == "density_m" or kind2 == "density_b")
        ):
        return np.where(
            kr < 1.e-3,
            1.-(kr**2)/6.,
            np.sin(kr)/kr
        )
    elif ((kind1 == "v_parallel") and (kind2 == "v_parallel")):
        return np.where(
            kr < 1.e-3,
            1.-3.*(kr**2)/10.,
            (3.*(kr**2-2.)*np.sin(kr)+6.*kr*np.cos(kr))/kr**3
        )
    elif ((kind1 == "v_perp") and (kind2 == "v_perp")):
        return np.where(
            kr < 1.e-3,
            1.-(kr**2)/10.,
            3.*(np.sin(kr)-kr*np.cos(kr))/kr**3
        )
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
        return np.where(
            kr < 1.e-3,
            kr/np.sqrt(3.),
            np.sqrt(3.)*(np.sin(kr)-kr*np.cos(kr))/kr**2
        )
    else:
        return 0.

def top_hat_window_function(kr):
    """
    Compute top-hat window function in real space.
    
    Parameters
    ----------
    kr: float or np.ndarray
        The argument of the window function.
    
    Returns
    -------
    float or np.ndarray
        Top-hat window function.
    """

    return np.where(
        kr < 1.e-3,
        1.-(kr**2)/10.,
        3.*(np.sin(kr)-kr*np.cos(kr))/kr**3
    )