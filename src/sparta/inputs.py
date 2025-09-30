"Module for defining the inputs of SPaRTA."

import numpy as np
import scipy.integrate as intg
from classy import Class
from scipy.interpolate import interp1d

#%% Define some global parameters
Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
nu_Lya = 2.47e15 # Lya frequency in Hz
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
Tcmb0 = 2.728 # CMB temperature in Kelvin
m_H = 1.6735575e-27 # Hydrogen atom mass in kg

#%% Class for setting simulation parameters

class SIM_PARAMS():
    """
    Class for setting simulation parameters.
    
    Parameters
    ----------
    sim_params: dictionary
        Contains the following simulation parameters:
            - z_abs: float
                The redshift where the photon was absorbed.
            - N: int 
                Number of photons to simulate.
            - Delta_L: float
                "Grid" resolution in Mpc.
            - x_stop: float
                Distance from absorption point to stop the simulation in units of the diffusion scale.
            - INCLUDE_VELOCITIES: bool
                If True, bulk peculiar velocities will be included in the simulation.
                Otherwise, assumes zero velocity throughout the simulation.
            - NO_CORRELATIONS: bool
                If True, the correlations in the velocitiy field will be ignored and at each sample a
                random velocity vector will be drawn independently from the previous velocity vector.
            - USE_INTERPOLATION_TABLES: bool
                If True, interpolation tables for velocity rms and correlation coefficients will be used,
                Otherwise, the rms and correlation coefficients will be computed from their integral definitions
                every time they are needed (MUCH more slowly that way).
            - INCLUDE_TEMPERATUR:, bool
                If True, non-zero Temperature will be accounted, based on its value in cosmo_params.
                Otherwise, zero temperature is assumed.
            - ANISOTROPIC_SCATTERING: bool
                If True, mu is drawn from a non-uniformal distribution (see Eq. 20 in arXiv: 2311.03447).
                Otherwise, it is drawn from a uniformal distribution.
            - INCLUDE_RECOIL: bool
                If True, atom recoil will be included during a scattering event (see Eq. 23 in arXiv: 2311.03447).
                Otherwise, this effect is not included and the outgoing photon's frequency is the same as 
                the incoming photon's frequency, in the atom rest frame.
            - STRAIGHT_LINE: bool
                If True, no scattering will happen and the photon will travel in a straight line.
                Otherwise, scattering are allowed.
            - CROSS_SECTION: string
                Type of cross-section to be used. Options are (default is 'Lorenzian'):
                    - 'LORENTZIAN': A simple Lorentzian profile is used when T=0. 
                    For a finite temperature, the Lorentzian profile is convolved 
                    with a Gaussian to yield to Voigt profile.
                    - 'PEEBLES': Cross-section from Peebles, P.J.E. (1993) Principles 
                    of Physical Cosmology (see also Eq. 49 in arXiv: 1704.03416).
                    Only works for T=0.
    
    """
    
    def __init__(self,sim_params):
        for k, v in sim_params.items():
            setattr(self, k, v)
        if not "x_stop" in  sim_params.keys():
            self.x_stop = None

#%% Class for setting cosmological parameters 
#   (and other variables/function with respect to the cosmological parameters)

class COSMO_PARAMS():
    """
    Class for setting cosmological parameters.
    
    Parameters
    ----------
    cosmo_params: dictionary
        Contains the following cosmological parameters:
            - h: float
                Hubble constant (in 100 km/sec/Mpc).
            - Omega_m: float
                Current matter density parameter.
            - Omega_b: float
                Current baryon density parameter.
            - A_s: float
                Amplitude of primordial curvator fluctuations 
                in the pivot scale 0.05 Mpc^-1.
            - n_s: float
                Spectral tilt of primordial curvator fluctuations.
            - T: float
                Temperature of the IGM in the simulation, in Kelvin.
            - x_HI: float
                Fraction of neutral hydrogen in the simulation.
    """
    
    def __init__(self,cosmo_params):
        for k, v in cosmo_params.items():
            setattr(self, k, v)
            
        # Compute other cosmological parameters
        self.H0 = 100*self.h # Hubble constant in km/sec/Mpc
        self.Omega_c = self.Omega_m - self.Omega_b # CDM portion
        self.rho_crit = 1.8788e-26 * self.h**2 # Critical energy density in kg/m^3
        self.rho_b0 = self.rho_crit*self.Omega_b # Baryon energy density in kg/m^3
    
    def run_CLASS(self):
        """
        Run CLASS with the input cosmological parameters.
        
        """
        CLASS_params = {}
        CLASS_params['h'] = self.h
        CLASS_params['Omega_cdm'] = self.Omega_c
        CLASS_params['Omega_b'] = self.Omega_b
        CLASS_params['A_s'] = self.A_s
        CLASS_params['n_s'] = self.n_s
        CLASS_params['T_cmb'] = Tcmb0
        CLASS_params['m_ncdm'] = "0.06" # eV
        CLASS_params['N_ncdm'] = 1
        CLASS_params['N_ur'] = 2.0308
        CLASS_params['output'] = 'mTk,vTk'
        CLASS_params['z_pk'] = 60.
        CLASS_params['gauge']='Newtonian'
        CLASS_params['P_k_max_1/Mpc'] = 100. # A rather high value for computing correlation at small distances
        # Run CLASS
        CLASS_OUTPUT = Class()
        CLASS_OUTPUT.set(CLASS_params)
        CLASS_OUTPUT.compute()
        self.CLASS_OUTPUT = CLASS_OUTPUT
        # Helium mass fraction (rho_He/rho_b)
        self.YHe = CLASS_OUTPUT.get_current_derived_parameters(['YHe'])['YHe']
        # Hydrogen number density at z=0
        self.n_H_z0 = (1.-self.YHe)*self.rho_b0/m_H # m^-3
        # Mean baryon mass. It is assumed that helium is doubly ionized when hydrogen is ionized
        self.m_b = m_H * (1./((2.-self.x_HI)*(1.-self.YHe)+(1.+2.*(1.-self.x_HI))*self.YHe/4.)) # kg
        
    def Delta_nu_star(self,z_abs):
        """
        Compute the frequency shift (in units of Lyman alpha frequncy) between 
        the absorption point and the point in which a photon that is escaping
        to infinity sees an optical depth of 1.
        
        Parameters
        ----------
        z_abs: float
            Redshift of absorption.
        
        Returns
        -------
        float:
            The frequency shift Delta_nu_star as given by Eq. (8) 
            in arXiv: astro-ph/9902180 (assuming matter domination). 
        """
        
        nu_star = 3.*c**3*A_alpha**2*self.n_H_z0*self.x_HI*pow(1.+z_abs,3./2.) # Hz^5
        nu_star /= 32.*np.pi**3 *nu_Lya**4 * (1000.*self.H0/Mpc_to_meter)*np.sqrt(self.Omega_m) # dimensionless
        return nu_star
    
    def R_SL(self,z1,z2):
        """
        Compute comoving straight line distance between z1 and z2.
        
        Parameters
        ----------
        z1: float
            Initial redshift.
        z2: float
            Final redshift.
        
        Returns
        -------
        float:
            The comoving distance between z1 and z2, in Mpc.
            Based on Eq. A3 in arXiv: 2101.01777 (This is int{c/H(z)} while
            assuming matter domination). 
        """
        
        r = abs(2.*c/1000./(self.H0*np.sqrt(self.Omega_m))*(1./np.sqrt(1+z1)-1./np.sqrt(1+z2))) # Mpc
        return r

    def R_SL_inverse(self,z1,r):
        """
        Compute the redshift that is located a comoving distance r fron z1
        (assuming z2 > z1).
        
        Parameters
        ----------
        z1: float
            Initial redshift.
        r: float
            Comoving distance, in Mpc.
        
        Returns
        -------
        float:
            The next redshift. 
        """
        z2 = pow(1./np.sqrt(1+z1)-1000.*self.H0*np.sqrt(self.Omega_m)*r/(2.*c),-2)-1. # Mpc
        return z2
    
    def r_star(self,z_abs):
        """
        Compute the comoving diffustion scale at z_abs.
        
        Parameters
        ----------
        z_abs: float
            Redshift of absorption.
        
        Returns
        -------
        float:
            The comoving diffusion scale in units of Mpc. This corrsponds to 
            the comoving distance that the photon has to travel (backwards in
            time) from the absorption point to the point from which the photon
            sees an optical depth of unity when it escapes to infinity. 
        """
        
        # What we have below is actually (z_star-z_abs)/(1+z_abs) = (nu_star-nu_Lya)/nu_Lya
        Delta_z_star = self.Delta_nu_star(z_abs) # dimensionless
        # Return output
        return self.R_SL(z_abs,z_abs + (1.+z_abs)*Delta_z_star) # Mpc 

    def compute_RMS(self,z,r):
        """
        Compute 1D velocity rms at redshift z, smoothed on a scale r.
        
        Parameters
        ----------
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
        k_CLASS = self.CLASS_OUTPUT.get_transfer(z=z)['k (h/Mpc)'][:]*self.h # 1/Mpc
        Transfer_z = self.CLASS_OUTPUT.get_transfer(z=z)
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
        Delta_R_sq = self.A_s*pow(k_arr/0.05,self.n_s-1.) # dimensionless
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
    
    def compute_2_point_correlation(self,z1,z2,v1_1D_rms,v2_1D_rms,r=None):
        """
        Compute velocity correlation coefficients for parallel 
        and perpendicular components, between two points at redshifts 
        z1 and z2.
        
        Parameters
        ----------
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
        k_CLASS = self.CLASS_OUTPUT.get_transfer(z=z1)['k (h/Mpc)'][:]*self.h # 1/Mpc
        Transfer_z1 = self.CLASS_OUTPUT.get_transfer(z=z1)
        Transfer_z2 = self.CLASS_OUTPUT.get_transfer(z=z2)
        theta_b_z1_CLASS = Transfer_z1['t_b'][:] # 1/Mpc
        theta_b_z2_CLASS = Transfer_z2['t_b'][:] # 1/Mpc
        # Comoving distance between z1 and z2
        if r is None:
            r = self.R_SL(z1,z2) # Mpc
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
        Delta_R_sq = self.A_s*pow(k_arr/0.05,self.n_s-1.)
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