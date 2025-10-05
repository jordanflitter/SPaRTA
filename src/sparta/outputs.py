"Module for defining the outputs of SPaRTA."

import numpy as np
from numpy.linalg import norm
from . import correlations, plotting
from .Lyman_alpha import compute_Lya_cross_section
from .interpolation import INTERPOLATOR

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
                 CLASS_OUTPUT = None,
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
        self.CLASS_OUTPUT = CLASS_OUTPUT if not sim_params.USE_INTERPOLATION_TABLES else None
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
            CLASS_OUTPUT = self.CLASS_OUTPUT,
            velocity_vector = self.velocity_vector.copy() if not self.velocity_vector is None else None,
            position_vector = self.position_vector.copy(),
            rotation_matrix = self.rotation_matrix.copy(),
            apparent_frequency = self.apparent_frequency,
            velocity_1D_rms = self.velocity_1D_rms,
            interpolator = self.interpolator
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
                CLASS_OUTPUT = self.CLASS_OUTPUT,
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
                    CLASS_OUTPUT = self.CLASS_OUTPUT,
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
                 random_seed,
                 interpolator,
                 CLASS_OUTPUT
                 
    ):
        self.z_abs = z_abs_data.redshift
        self.z_abs_data = z_abs_data
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.points_data = [z_abs_data]
        self.random_seed = random_seed
        self.interpolator = interpolator
        self.CLASS_OUTPUT = CLASS_OUTPUT
    
    def append(self,point_data):
        """
        Append current point to this object.
        
        Parameters
        ----------
        point_data: :class:`~COSMO_POINT_DATA`
            New data point to append.
        """
        
        self.points_data.append(point_data)

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
        # Run CLASS
        self.CLASS_OUTPUT = cosmo_params.run_CLASS()
        cosmo_params.update_cosmo_params_with_CLASS(self.CLASS_OUTPUT)
        # Set interpolator
        self.interpolator = INTERPOLATOR(
            cosmo_params = cosmo_params,
            sim_params = sim_params,
            z_abs = self.z_abs,
            CLASS_OUTPUT = self.CLASS_OUTPUT
        )
        # Initialize interpolation tables for the velocity rms and Pearson coefficients
        if self.sim_params.INCLUDE_VELOCITIES and self.sim_params.USE_INTERPOLATION_TABLES:
            self.interpolator.initialize_velocity_interpolation_tables()
            # Destroy the CLASS_OUTPUT field (no need to save it after the velocity interpolation tables were initialized)
            self.CLASS_OUTPUT = None
        # Initialize interpolation tables for anisotropic scattering
        if self.sim_params.ANISOTROPIC_SCATTERING and not self.sim_params.STRAIGHT_LINE:
            self.interpolator.initialize_mu_distribution_tables()
    
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