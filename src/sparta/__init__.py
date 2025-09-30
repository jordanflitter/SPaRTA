"""
This is SPaRTA = speedy Lyman alpha ray tracing algorithm. 
SPaRTA works on a virtual "grid" and backwards in time, i.e. 
we begin from Lyman alpha frequency and track the photon's past trajectory. 
This is done that way because we can interepret each scattering point as a 
potential source of emission.
At each "cell" of the grid the commulative optical depth is updated, and 
once it crosses a random threshold (drawn from exp(-tau) distribution), 
the photon is scattered to a random direction (from an isotropic/anisotropic
angular distribution). 
The optical depth depends both on the temperature (a free parameter in the
simulation) and the apparent frequency that hydrogen atom in the IGM sees.
SPaRTA accounts for the redshift of the photon (or blueshift, since
we work backwards), and also for the bulk peculiar motion of gas particles
in the IGM. This is done by correlating the smoothed velocities at 
neighbouring cells and computing the projected relative velocity along the
photon's trajectory.
SPaRTA allows simulating an arbitrary number of photons, each of which
has its own special random seed. Once the simulation is over, the user can
plot the evolution of individual photons by tracking their progress in 
redshift, apparent frequency, and distance from the absorber. In addition,
scatter plots of the redshift of the emitter vs. the distance the photon has 
traveled (or vs. the relative velocity of the abosrber to the emitter) can
be plotted.
"""

from .inputs import (
    COSMO_PARAMS,
    SIM_PARAMS,
)
from .outputs import (
    COSMO_POINT_DATA,
    PHOTON_POINTS_DATA,
    ALL_PHOTONS_DATA
)
from .post_processing import (
    SIM_DATA,
    HISTOGRAM_DATA
)
from . cosmology import(
    compute_Lya_cross_section
)
from . misc import(
    compute_correlation_function
)
from . plotting import(
    plot_cross_section
)