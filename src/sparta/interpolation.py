"Module for performing interpolation."

import numpy as np
from scipy.interpolate import RectBivariateSpline
from . import correlations, Lyman_alpha

class INTERPOLATOR():
    """
    Class for storing the interpolation tables in the simulation.
    
    Parameters
    ----------
    cosmo_params: :class:`~COSMO_PARAMS`
        The cosmological parameters and functions for the simulation.
        Needs to be passed after CLASS was run.
    sim_params: :class:`~SIM_PARAMS`
        The simulation parameters.
    
    The class stores all the interpolation tables used in the simulation.
    """

    def __init__(self,
                 cosmo_params,
                 sim_params,
                 z_abs,
                 CLASS_OUTPUT,
                 interpolate_RMS = None,
                 interpolate_rho_parallel = None,
                 interpolate_rho_perp = None,
                 mu_table_core = None,
                 mu_table_wing = None
    ):
        
        self.cosmo_params = cosmo_params
        self.sim_params = sim_params
        self.z_abs = z_abs
        self.CLASS_OUTPUT = CLASS_OUTPUT if sim_params.USE_INTERPOLATION_TABLES else None
        self.interpolate_RMS = interpolate_RMS
        self.interpolate_rho_parallel = interpolate_rho_parallel
        self.interpolate_rho_perp = interpolate_rho_perp
        self.mu_table_core = mu_table_core
        self.mu_table_wing = mu_table_wing
    
    def initialize_velocity_interpolation_tables(self):
        """
        Initialize interpolation tables for the velocity rms and the Pearson coefficients.
        """
        
        # Create interpolation table for the velocity RMS
        z_end = (1.+self.z_abs)*self.sim_params.nu_stop-1.
        z_array = np.linspace(self.z_abs,1.02*z_end,150) # We take a slightly higher z_end because of Doppler shifts that brings us to higher redshifts
        rms_array = np.zeros_like(z_array)
        for zi_ind, zi in enumerate(z_array):
            rms_array[zi_ind] = correlations.compute_RMS(
                CLASS_OUTPUT = self.CLASS_OUTPUT,
                z = zi,
                r_smooth = self.sim_params.Delta_L,
                kind = "velocity"
            )
        # NOTE: why do I do 2D interpolation if the data is 1D?
        #       Apparently, 2D interpolation is more efficient! (very weird, I know)    
        self.interpolate_RMS = RectBivariateSpline(
                z_array,
                np.array([0,1,2,3]),
                np.repeat(rms_array, 4, axis=0).reshape(len(z_array),4)
            )
        # Create interpolation tables for the Pearson coefficient
        if not self.sim_params.NO_CORRELATIONS:
            r_array = np.linspace(0,10*self.sim_params.Delta_L,100) # Mpc
            r_array = np.append(r_array,self.sim_params.Delta_L) # Mpc
            r_array = np.unique(r_array) # Mpc
            r_array = np.sort(r_array) # Mpc
            rho_parallel_array = np.zeros_like(r_array)
            rho_perp_array = np.zeros_like(r_array)
            # The Pearson coefficient seems to be very weakly dependent on redshift
            # (relative differences of 1e-6 when 1-rho is examined!).
            # For setting the interpolation table, we need to choose an arbitrary redshift.
            # We choose z_abs.
            z_ = self.z_abs
            for r_ind, r in enumerate(r_array):
                rho_dict = correlations.compute_Pearson_coefficient(
                    CLASS_OUTPUT = self.CLASS_OUTPUT,
                    z1 = z_,
                    z2 = self.cosmo_params.R_SL_inverse(z_,r),
                    r = r,
                    r_smooth = self.sim_params.Delta_L,
                    kinds_list = [("v_parallel","v_parallel"), ("v_perp","v_perp")]
                )
                rho_parallel_array[r_ind] = rho_dict["v_parallel,v_parallel"]
                rho_perp_array[r_ind] = rho_dict["v_perp,v_perp"]
            # NOTE: why do I do 2D interpolation if the data is 1D?
            #       Apparently, 2D interpolation is more efficient! (very weird, I know)
            self.interpolate_rho_parallel = RectBivariateSpline(
                r_array,
                np.array([0,1,2,3]),
                np.repeat(rho_parallel_array, 4, axis=0).reshape(len(r_array),4)
            )
            self.interpolate_rho_perp = RectBivariateSpline(
                r_array,
                np.array([0,1,2,3]),
                np.repeat(rho_perp_array, 4, axis=0).reshape(len(r_array),4)
            )
            # Destroy the CLASS_OUTPUT field (no need to save it after the velocity interpolation tables were initialized)
            self.CLASS_OUTPUT = None
    
    def initialize_mu_distribution_tables(self):
        """
        Initialize interpolation tables for the inverse mu CDF, according to Eq. (20)
        in arXiv: 2311.03447.
        """
        
        self.mu_table_core = Lyman_alpha.inverse_mu_CDF(11./24.,3./24.)
        self.mu_table_wing = Lyman_alpha.inverse_mu_CDF(3./8.,3./8.)