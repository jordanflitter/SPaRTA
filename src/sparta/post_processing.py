"Module for post processing the outputs of SPaRTA."

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import scipy.integrate as intg
from scipy.optimize import curve_fit
from scipy.stats import norm as norm_dist
from scipy.stats import beta as beta_dist

#%% Class for collecting data from the simulation

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