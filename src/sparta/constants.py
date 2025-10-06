"Module for defining the constants used in SPaRTA."

Mpc_to_meter = 3.085677581282e22
c = 2.99792458e8 # Speed of light in m/sec
h_P = 6.62606896e-34 # Planck Constant in J*sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
nu_Lya = 2.47e15 # Lya frequency in Hz
lambda_alpha = c/nu_Lya # Lya wavelength in m
A_alpha = 6.25e8 # Spontaneous decay rate of hydrogen atom from the 2p state to the 1s state in Hz
Tcmb0 = 2.728 # CMB temperature in Kelvin
m_H = 1.6735575e-27 # Hydrogen atom mass in kg
A_alpha_dimensionless = A_alpha/nu_Lya
Lyman_beta = 32./27. # Lyb frequency in units of Lya frequency