"Module for defining the outputs of SPaRTA."

import numpy as np
import tqdm
from numpy.linalg import norm
from . import correlations, plotting
from .Lyman_alpha import compute_Lya_cross_section, draw_from_voigt_distribution
from .interpolation import INTERPOLATOR

#%% Define some global parameters
Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
h_P = 6.62606896e-34 # Planck Constant in J*sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya
Lyman_beta = 32./27. # Lyb frequency in units of Lya frequency

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
                 velocity_1D_rms = None,
                 interpolator = None,
    ):
        self.redshift = redshift
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.velocity_vector = velocity_vector
        self.position_vector = position_vector
        self.rotation_matrix = rotation_matrix
        self.apparent_frequency = apparent_frequency
        self.velocity_1D_rms = velocity_1D_rms
        self.interpolator = interpolator
    
    def copy(self):
        """
        Copy the content of this point to another object.
        
        Returns
        -------
        :class:`~COSMO_POINT_DATA`
            A copy of the content of this point
        """
        
        return COSMO_POINT_DATA(
            redshift=self.redshift,
            cosmo_params=self.cosmo_params,
            sim_params=self.sim_params,
            velocity_vector=self.velocity_vector.copy() if not self.velocity_vector is None else None,
            position_vector=self.position_vector.copy(),
            rotation_matrix=self.rotation_matrix.copy(),
            apparent_frequency=self.apparent_frequency,
            velocity_1D_rms=self.velocity_1D_rms,
            interpolator=self.interpolator
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
    
    def evaluate_RMS(self):
        """
        Compute the smoothed 1D velocity rms for this point. 
        Interpolation is performed if USE_INTERPOLATION_TABLES = True.
        """
        not_computed = True
        # Interpolate!
        if self.sim_params.USE_INTERPOLATION_TABLES:
            try:
                self.velocity_1D_rms = self.interpolator.interpolate_RMS(self.redshift,0)[0,0]
                not_computed = False
            except ValueError: # Interpolation can fail because of Doppler shifts that brings us to higher redshifts
                not_computed = True
        # Integrate!
        elif not_computed:
            self.velocity_1D_rms = correlations.compute_RMS(
                CLASS_OUTPUT = self.cosmo_params.CLASS_OUTPUT,
                z = self.redshift,
                r_smooth = self.sim_params.Delta_L,
                kind = "velocity"
            )
        
    def evaluate_Pearson_coefficient(self,z1_data,r=None):
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
            if r is None:
                r = self.cosmo_params.R_SL(z1_data.redshift,self.redshift)
            # Interpolate!
            if self.sim_params.USE_INTERPOLATION_TABLES:
                self.rho_v_parallel = self.interpolator.interpolate_rho_parallel(r,0)[0,0]
                self.rho_v_perp = self.interpolator.interpolate_rho_perp(r,0)[0,0]

                # Sanity check: -1 <= rho <= 1
                if self.rho_v_parallel**2 > 1.:
                    print(f"Warning: At (z1,z2)={z1_data.redshift,self.redshift} the correlation coefficient for v_parallel is rho={self.rho_v_parallel}")
                if self.rho_v_perp**2 > 1.:
                    print(f"Warning: At (z1,z2)={z1_data.redshift,self.redshift} the correlation coefficient for v_perp is rho={self.rho_v_perp}")
            # Integrate!
            else:
                rho_dict = correlations.compute_Pearson_coefficient(
                    CLASS_OUTPUT = self.cosmo_params.CLASS_OUTPUT,
                    z1 = z1_data.redshift,
                    z2 = self.redshift,
                    r = r,
                    r_smooth = self.sim_params.Delta_L,
                    kinds_list = [("v_parallel","v_parallel"), ("v_perp","v_perp")]
                )
                self.rho_v_parallel = rho_dict["v_parallel,v_parallel"]
                self.rho_v_perp = rho_dict["v_perp,v_perp"]
        else:
            self.rho_v_parallel = 0.
            self.rho_v_perp = 0.
    
    def draw_conditional_velocity_vector(self,z1_data,r=None):
        """
        Draw a conditional velocity vector for this point based on the 
        velocity vector of the previous sample.
        
        Parameters
        ----------
        z1_data: :class:`~COSMO_POINT_DATA`
            The data of the previous point.
        r: float, optional
            Comoving distance between the two points, in Mpc.
        """

        # Compute the Pearson coefficients for the parallel and 
        # perpendicular components of the velocity field.
        self.evaluate_Pearson_coefficient(z1_data,r)
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
        # This is the integrand for the tau integral.
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
            # NOTE: here we do not divide the scale by sqrt(2) as we do in simulate_one_photon.
            #       This is because we are looking for the relative thermal velocity, so the variance is two times larger
            #       (since the thermal velocities are not correlated)
            v_thermal_rel_parallel = np.random.normal(scale=self.cosmo_params.Delta_nu_D)
            z_ini_data.apparent_frequency /= (1.-v_thermal_rel_parallel)
        # Draw the position vector from uncorrelated Gaussian distributions
        tilde_nu = np.abs(z_ini_data.apparent_frequency-1.)/self.cosmo_params.Delta_nu_star(self.z_abs) # dimensionless
        scale = np.sqrt(2./9.*tilde_nu**3)*self.cosmo_params.r_star(self.z_abs) # Mpc
        z_ini_data.position_vector = np.random.normal(scale=scale,size=3) # Mpc
        # Correct initial frequency due to peculiar velocity
        # NOTE: it would be more consistent to draw the poisition vector after the frequency was corrected due to peculiar velocities.
        # However, the distamce from the origin is expected to be small, so peculiar relative velocities are very small and barely matter.
        # We do it like this in order to use the distance from the origin for the calculation of the Pearson coefficient (and for drawing 
        # the conditional velocity vector)
        if self.sim_params.INCLUDE_VELOCITIES:
            # Compute the smoothed 1D velocity RMS in z_ini
            z_ini_data.evaluate_RMS()
            # Draw velocity at z_ini based on the velocity vector at z_abs
            z_ini_data.draw_conditional_velocity_vector(z_abs_data,r=norm(z_ini_data.position_vector))
            # Compute parallel component of relative velocity with respect to the last point
            # Remember: the first component in our velocity vector is always aligned with the photon's trajectory
            v_rel_parallel = z_ini_data.velocity_vector[0] - z_abs_data.velocity_vector[0] # dimensionless
            # Correct initial frequency due to peculiar velocity
            z_ini_data.apparent_frequency /= (1.-v_rel_parallel)
        # Return output
        return z_ini_data
        
    def plot_photon_trajectory(
        self,
        scale=5.,
        ax = None,
        **kwargs
    ):
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
        
        return plotting.plot_photon_trajectory(
            photon_points_data = self,
            scale = scale,
            ax = ax,
            **kwargs
        )
    
    def plot_apparent_frequency(
        self,
        ax = None,
        **kwargs
    ):
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
        
        return plotting.plot_apparent_frequency(
            photon_points_data = self,
            ax = ax,
            **kwargs
        )

    def plot_distance(
        self,
        intermediate_pts = True,
        ax = None,
        **kwargs
    ):
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
        
        return plotting.plot_distance(
            photon_points_data = self,
            intermediate_pts = intermediate_pts,
            ax = ax,
            **kwargs
        )

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
        # Set interpolator
        self.interpolator = INTERPOLATOR(
            cosmo_params = cosmo_params,
            sim_params = sim_params,
            z_abs = self.z_abs,
            nu_stop = self.nu_stop
        )
        # Initialize interpolation tables for the velocity rms and Pearson coefficients
        if self.sim_params.INCLUDE_VELOCITIES and self.sim_params.USE_INTERPOLATION_TABLES:
            self.interpolator.initialize_velocity_interpolation_tables()
        # Initialize interpolation tables for anisotropic scattering
        if self.sim_params.ANISOTROPIC_SCATTERING and not self.sim_params.STRAIGHT_LINE:
            self.interpolator.make_mu_distribution_tables()
    
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
            if self.sim_params.USE_INTERPOLATION_TABLES:
                z_abs_data.interpolator = self.interpolator
            # Compute the smoothed 1D velocity RMS in z_abs
            z_abs_data.evaluate_RMS()
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
                # NOTE: We don't need to track the photon's position at every z',
                # and instead we keep it to be in z_prime_old (which was set to
                # be at z_i)
                next_redshift = self.cosmo_params.R_SL_inverse(z_prime_data.redshift,self.sim_params.Delta_L)
                z_prime_data.redshift = next_redshift
                if self.sim_params.INCLUDE_VELOCITIES:
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
                    if abs(z_i_data.apparent_frequency-1.) < 0.2*self.cosmo_params.Delta_nu_D:
                        mu_rnd = self.interpolator.mu_table_core(np.array([np.random.rand()]))[0]
                    else:
                        mu_rnd = self.interpolator.mu_table_wing(np.array([np.random.rand()]))[0]
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
                    # Note that in our dimensionless units, Delta_nu_D also equals the rms of thermal velocity
                    # Also note that we only need two components for the thermal velocity, not three
                    v_thermal_perp = np.random.normal(scale=self.cosmo_params.Delta_nu_D/np.sqrt(2.)) # dimensionless
                    if self.cosmo_params.T > 0.:
                        v_thermal_parallel = (
                            self.cosmo_params.Delta_nu_D * 
                            draw_from_voigt_distribution(
                                (z_i_data.apparent_frequency-1.)/self.cosmo_params.Delta_nu_D,
                                self.cosmo_params.a_T
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