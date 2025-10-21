# SPaRTA

This is `SPaRTA` 🛡️ = SPeedy Lyman alpha Ray Tracing Algorithm. The code can be used to perform quickly MC simulations of the trajectories of absorbed Lyman alpha photons in the IGM, plotting them, and gather insights on their properties.

An example of five photon trajectories is given below. On largest scales they appear to travel in straight-lines, while on small scales they make a random walk with a decreasing step size.

![Trajectories](https://github.com/jordanflitter/SPaRTA/blob/main/images/Trajectories.png)

When simulating many photons, their statistics can be studied. This includes for example the radial distributions from the point of absorption, and the window function (Fourier transform of the radial distributions).

![Distributions](https://github.com/jordanflitter/SPaRTA/blob/main/images/Distributions.png)

## Installation
The installation of SPaRTA is very easy. In your terminal application, type:
```
git clone git@github.com:jordanflitter/SPaRTA.git
```
This will clone the `SPaRTA` repository to your current directory. Then run the following commands to complete the installation.
```
cd SPaRTA
pip install .
```

## Usage
Check out the [tutorials](https://github.com/jordanflitter/SPaRTA/tree/main/Tutorials) to learn how to use the code.
