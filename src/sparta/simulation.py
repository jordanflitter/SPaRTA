"Module for running the simulation in SPaRTA."

import numpy as np
import tqdm
from numpy.linalg import norm
from .Lyman_alpha import draw_from_voigt_distribution
from . outputs import COSMO_POINT_DATA, PHOTON_POINTS_DATA

#%% Define some global parameters
Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
h_P = 6.62606896e-34 # Planck Constant in J*sec
nu_Lya = 2.47e15 # Lya frequency in Hz
m_H = 1.6735575e-27 # Hydrogen atom mass in kg

def draw_first_point(photon_data):
    """
    Draw the first point for a particular photon.
    This is done from the analytical fit of Loeb & Rybicki 1999 
    (arXiv: astro-ph/9902180).
    
    Parameters
    ----------
    photon_data: :class:`~PHOTON_POINTS_DATA`
        Object that will contain all the data points for this photon.
    
    Returns
    -------
    point_data: :class:`~COSMO_POINT_DATA`
        Data of the first point.
    """
    
    # Copy the content of z_abs_data into z_ini_data.
    z_ini_data = photon_data.z_abs_data.copy()
    # Find the next redshift from the initial frequency shift
    z_ini = (1.+photon_data.z_abs)*(1.+ photon_data.sim_params.Delta_nu_initial) - 1.
    z_ini_data.redshift = z_ini
    # Define initial frequency at z_ini
    z_ini_data.apparent_frequency = (1.+z_ini)/(1.+photon_data.z_abs) # dimensionless (in units of Lya frequency)
    # Correct initial frequency due to temperature
    if photon_data.sim_params.INCLUDE_TEMPERATURE:
        # Draw a random thermal velocity from Gaussian distribution
        # NOTE: here we do not divide the scale by sqrt(2) as we do in simulate_one_photon.
        #       This is because we are looking for the relative thermal velocity, so the variance is two times larger
        #       (since the thermal velocities are not correlated)
        v_thermal_rel_parallel = np.random.normal(scale=photon_data.cosmo_params.Delta_nu_D)
        z_ini_data.apparent_frequency /= (1.-v_thermal_rel_parallel)
    # Draw the position vector from uncorrelated Gaussian distributions
    tilde_nu = np.abs(z_ini_data.apparent_frequency-1.)/photon_data.cosmo_params.Delta_nu_star(photon_data.z_abs) # dimensionless
    scale = np.sqrt(2./9.*tilde_nu**3)*photon_data.cosmo_params.r_star(photon_data.z_abs) # Mpc
    z_ini_data.position_vector = np.random.normal(scale=scale,size=3) # Mpc
    # Correct initial frequency due to peculiar velocity
    # NOTE: it would be more consistent to draw the poisition vector after the frequency was corrected due to peculiar velocities.
    # However, the distamce from the origin is expected to be small, so peculiar relative velocities are very small and barely matter.
    # We do it like this in order to use the distance from the origin for the calculation of the Pearson coefficient (and for drawing 
    # the conditional velocity vector)
    if photon_data.sim_params.INCLUDE_VELOCITIES:
        # Compute the smoothed 1D velocity RMS in z_ini
        z_ini_data.evaluate_RMS()
        # Draw velocity at z_ini based on the velocity vector at z_abs
        z_ini_data.draw_conditional_velocity_vector(photon_data.z_abs_data,r=norm(z_ini_data.position_vector))
        # Compute parallel component of relative velocity with respect to the last point
        # Remember: the first component in our velocity vector is always aligned with the photon's trajectory
        v_rel_parallel = z_ini_data.velocity_vector[0] - photon_data.z_abs_data.velocity_vector[0] # dimensionless
        # Correct initial frequency due to peculiar velocity
        z_ini_data.apparent_frequency /= (1.-v_rel_parallel)
    # Return output
    return z_ini_data

def simulate_one_photon(photon_data,random_seed):
    """
    Simulate one photon, given a random seed.
    
    Parameters
    ----------
    photon_data: :class:`~PHOTON_POINTS_DATA`
        Object that will contain all the data points for this photon.
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
    if photon_data.sim_params.INCLUDE_VELOCITIES:
        # Draw a random velocity at z_abs.
        # Velocities are dimensionless in this code as they are normalized by c.
        photon_data.z_abs_data.velocity_vector = np.random.normal(scale=photon_data.z_abs_data.velocity_1D_rms,size=3) # dimensionless
    # Draw the first position of the photon outside the origin.
    # This is the diffusion regime, where analytical result can be used.
    if not photon_data.sim_params.STRAIGHT_LINE:
        z_ini_data = draw_first_point(photon_data)
        photon_data.append(z_ini_data.copy())
        # Initialize z_i to be at z_ini
        # Each z_i corresponds to a new scattering point
        z_i_data = z_ini_data.copy()
    else:
        z_i_data = photon_data.z_abs_data.copy()
    # Compute tau integrand at z_ini
    dtau_2_dL_curr = z_i_data.compute_dtau_2_dL() # 1/m
    # Scatter the photon until we reached final frequency
    while z_i_data.apparent_frequency < photon_data.sim_params.nu_stop:
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
        while tau_integral < tau_rnd and z_prime_data.apparent_frequency < photon_data.sim_params.nu_stop:
            # Update tau integrand
            dtau_2_dL_prev = dtau_2_dL_curr # 1/m
            # Set z_prime_old to be z_prime
            # We will use the velocity in z_prime_old to draw a correlated
            # velocity at the new z_prime
            z_prime_old_data = z_prime_data.copy()
            # Update z_prime
            # NOTE: We don't need to track the photon's position at every z',
            # and instead we keep it to be in z_prime_old (which was set to
            # be at z_i)
            next_redshift = photon_data.cosmo_params.R_SL_inverse(z_prime_data.redshift,photon_data.sim_params.Delta_L)
            z_prime_data.redshift = next_redshift
            if photon_data.sim_params.INCLUDE_VELOCITIES:
                # Update the smoothed 1D velocity RMS in z'
                z_prime_data.evaluate_RMS()
                # Draw velocity at z_prime based on the velocity vector at z_prime_old
                z_prime_data.draw_conditional_velocity_vector(z_prime_old_data)
                # Compute parallel component of relative velocity with respect to the last point
                # Remember: the first component in our velocity vector is always aligned with the photon's trajectory
                v_rel_parallel = z_prime_data.velocity_vector[0] - z_prime_old_data.velocity_vector[0] # dimensionless
            # Calculate apparent frequency at z_prime. 
            # First, we blueshift the apparent frequency from previous redshift
            z_prime_data.apparent_frequency = z_prime_old_data.apparent_frequency*(1.+z_prime_data.redshift)/(1.+z_prime_old_data.redshift) # dimensionless
            if photon_data.sim_params.INCLUDE_VELOCITIES:
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
            dtau = dtau_2_dL * photon_data.sim_params.Delta_L * Mpc_to_meter # dimensionless
            tau_integral += dtau  # dimensionless
        # Draw a random direction in which the photon has propagated
        if photon_data.sim_params.STRAIGHT_LINE:
            mu_rnd = 1.
            phi_rnd = 0.
        else:
            if photon_data.sim_params.ANISOTROPIC_SCATTERING:
                # Draw random mu from the phase function,
                # given by Eq. (20) in arXiv: 2311.03447
                if abs(z_i_data.apparent_frequency-1.) < 0.2*photon_data.cosmo_params.Delta_nu_D:
                    mu_rnd = photon_data.interpolator.mu_table_core(np.array([np.random.rand()]))[0]
                else:
                    mu_rnd = photon_data.interpolator.mu_table_wing(np.array([np.random.rand()]))[0]
            else:
                # Draw random mu from a uniform distribution
                mu_rnd = -1.+2.*np.random.rand()
            # phi is always drawn from a uniform distribution
            phi_rnd = 2.*np.pi*np.random.rand()
        # Compute comoving distance between z_i and z_{i+1} (assumed to be z_prime),
        # according to the straight line formula
        L_i = photon_data.cosmo_params.R_SL(z_i_data.redshift,z_prime_data.redshift) # Mpc
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
            if photon_data.sim_params.INCLUDE_VELOCITIES:
                z_i_data.rotate(mu_rnd,phi_rnd)
            # Change frequency of scattered photon due to recoil
            if photon_data.sim_params.INCLUDE_RECOIL and photon_data.sim_params.INCLUDE_TEMPERATURE:
                # Draw a random thermal velocity vector. The perpendicular component is drawn from a normal distribution
                # while the parallel component is drawn from Eq. (25) in arXiv: 2311.03447.
                # Note that in our dimensionless units, Delta_nu_D also equals the rms of thermal velocity
                # Also note that we only need two components for the thermal velocity, not three
                v_thermal_perp = np.random.normal(scale=photon_data.cosmo_params.Delta_nu_D/np.sqrt(2.)) # dimensionless
                if photon_data.cosmo_params.T > 0.:
                    v_thermal_parallel = (
                        photon_data.cosmo_params.Delta_nu_D * 
                        draw_from_voigt_distribution(
                            (z_i_data.apparent_frequency-1.)/photon_data.cosmo_params.Delta_nu_D,
                            photon_data.cosmo_params.a_T
                        )
                    ) # dimensionless
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
    return

def simulate_N_photons(all_photons_data,random_seed=None):
    """
    Simulate many photons.
    
    Many different photons are simulated, each of which with a different
    random seed (incremented by one at the end of each simulation). 
    The total number of simulated photons is determined by self.sim_params.N.

    Parameters
    ----------
    all_photons_data: :class:`~ALL_PHOTONS_DATA`
        Object that will contain all the data from all the photons in the simulation.
    random_seed: int, optional
        The random seed for the first photon in the simulation. Default is z_abs.
    """
    
    if random_seed is None:
            random_seed = int(all_photons_data.sim_params.z_abs)

    # Initialize a photon in z_abs with Lyman alpha frequency (1 
    # in our units, since we normalize frequency by nu_Lya)
    z_abs_data = COSMO_POINT_DATA(
        redshift=all_photons_data.z_abs,
        cosmo_params=all_photons_data.cosmo_params,
        sim_params=all_photons_data.sim_params,
        apparent_frequency=1.,
        interpolator = all_photons_data.interpolator,
        CLASS_OUTPUT = all_photons_data.CLASS_OUTPUT
    )
    # Compute the smoothed 1D velocity RMS in z_abs
    if all_photons_data.sim_params.INCLUDE_VELOCITIES:
        z_abs_data.evaluate_RMS()
    
    # Simulate N photons
    for n  in tqdm.tqdm(range(all_photons_data.sim_params.N),
                        desc="",
                        unit="photons",
                        disable=False,
                        total= all_photons_data.sim_params.N):
        # Create a photon data object and initialize it with the point at z_abs
        photon_data = PHOTON_POINTS_DATA(
            z_abs_data = z_abs_data.copy(),
            cosmo_params = all_photons_data.cosmo_params,
            sim_params = all_photons_data.sim_params,
            random_seed = random_seed,
            interpolator = all_photons_data.interpolator,
            CLASS_OUTPUT = all_photons_data.CLASS_OUTPUT
        )
        simulate_one_photon(photon_data,random_seed)
        # Append the data of this photon to the object
        all_photons_data.append(photon_data)
        # Use a different random seed for the next photon
        random_seed += 1