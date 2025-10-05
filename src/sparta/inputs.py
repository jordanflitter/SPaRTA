"Module for defining the inputs of SPaRTA."

import numpy as np
from classy import Class

#%% Define some global parameters
Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
Tcmb0 = 2.728 # CMB temperature in Kelvin
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya
Lyman_beta = 32./27. # Lyb frequency in units of Lya frequency

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
        
    def update_sim_params_with_cosmo_params(self,cosmo_params):
        """
        Update some of the simulation parameters with the cosmological parameters.
        
        Parameters
        ----------
        cosmo_params: :class:`~COSMO_PARAMS`
            The cosmological parameters and functions for the simulation.
        
        """
        
        if not self.x_stop is None: 
            r_stop = self.x_stop*cosmo_params.r_star(self.z_abs) # Mpc
            z_stop = self.cosmo_params.R_SL_inverse(self.z_abs,r_stop) # final redshift
            self.nu_stop = (1.+z_stop)/(1.+self.z_abs) # frequency to stop the simulation (in units of Lya frequency)
            if self.nu_stop > Lyman_beta:
                self.nu_stop = Lyman_beta
        else:
            self.nu_stop = Lyman_beta
            z_stop = (1.+self.z_abs)*self.nu_stop - 1. # final redshift
            r_stop = cosmo_params.R_SL(self.z_abs,z_stop) # Mpc
            self.x_stop = r_stop/cosmo_params.r_star(self.z_abs) # dimensionless
        

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
        self.YHe = 0.2455 # Helium mass fraction (rho_He/rho_b)
        self.n_H_z0 = (1.-self.YHe)*self.rho_b0/m_H # # Hydrogen number density at z=0 in m^-3
        self.m_b = m_H * (1./((2.-self.x_HI)*(1.-self.YHe)+(1.+2.*(1.-self.x_HI))*self.YHe/4.)) # # Mean baryon mass in kg. It is assumed that helium is doubly ionized when hydrogen is ionized
        # Voigt profile parameters
        self.Delta_nu_D = np.sqrt(2*k_B*self.T/self.m_b/c**2) # dimensionless (in units of nu_Lya)
        self.a_T = A_alpha_dimensionless/4/np.pi/self.Delta_nu_D # dimensionless
        
    
    def run_CLASS(self):
        """
        Run CLASS with the input cosmological parameters.
        
        Returns
        -------
        CLASS_OUTPUT: :class:`classy.Class`
            An object containing all the information from the CLASS calculation.
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
        CLASS_params['output'] = 'mTk,vTk,mPk'
        CLASS_params['z_pk'] = 60.
        CLASS_params['gauge']='Newtonian'
        CLASS_params['P_k_max_1/Mpc'] = 100. # A rather high value for computing correlation at small distances
        # Run CLASS
        CLASS_OUTPUT = Class()
        CLASS_OUTPUT.set(CLASS_params)
        CLASS_OUTPUT.compute()
        # Return output
        return CLASS_OUTPUT
    
    def update_cosmo_params_with_CLASS(self,CLASS_OUTPUT):
        """
        Update some of the cosmological parameters with output from CLASS (due to more precise calculation of Y_He).
        
        Parameters
        ----------
        CLASS_OUTPUT: :class:`classy.Class`
            An object containing all the information from the CLASS calculation.
        
        """
        # Helium mass fraction (rho_He/rho_b)
        self.YHe = CLASS_OUTPUT.get_current_derived_parameters(['YHe'])['YHe']
        # Hydrogen number density at z=0
        self.n_H_z0 = (1.-self.YHe)*self.rho_b0/m_H # m^-3
        # Mean baryon mass. It is assumed that helium is doubly ionized when hydrogen is ionized
        self.m_b = m_H * (1./((2.-self.x_HI)*(1.-self.YHe)+(1.+2.*(1.-self.x_HI))*self.YHe/4.)) # kg
        # Voigt profile parameters
        self.Delta_nu_D = np.sqrt(2*k_B*self.T/self.m_b/c**2) # dimensionless (in units of nu_Lya)
        self.a_T = A_alpha_dimensionless/4/np.pi/self.Delta_nu_D # dimensionless
        
        
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