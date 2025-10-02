# PSAND: Planetary Structure ANd Dynamics
A python method to couple planetary structure and dynamics, as published in Hallatt & Millholland (2025): https://arxiv.org/abs/2509.22923.

Data files for interpolation take up 117 MB (compressed) / 538 MB (uncompressed).

## A minimal working example

This example loads our interpolation functions assuming solar composition gas, and a 10 Earth mass core. We set the irradiation temperature to 900 K and the total planet mass to 250 Earth masses. We integrate equation 5 from Hallatt & Millholland (2025). We then plot the entropy as a function of time.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# define our constants (cgs units)
yr=31560000.0
kb=1.3807e-16
mh=1.6726e-24

# planet properties (fixed in time for this example)
Tirr=900.
M=250.

# load our interpolation functions for planet structure
fTdm=np.load('fTm_mc=10_z=0_1au.npy',allow_pickle=True).item()
fL=np.load('fL_mc=10_z=0_1au.npy',allow_pickle=True).item()
fR=np.load('fR_mc=10_z=0_1au.npy',allow_pickle=True).item()

# this function defines our extra heating rate. for this example, we set it to zero.
def Lx(t):
  return 0.

# this function evolves planet entropy by "stepping through the adiabats" (equation 5 of Hallatt & Millholland (2025)).
def dSdtm(S,Lextra_,M,Tirr):
  return (-fL(S,M,Tirr)+Lextra_)/fTdm(S,M,Tirr)

# this function yields the differential equation to be solved with scipy
def fevo(t,y):
  S=y[0]
  Lextra=Lx(t)
  dsdt=dSdtm(S,Lextra,M,Tirr)
  return [dsdt]

# define initial conditions
t0,tend=1e7*yr,1e10*yr
S0=10.
sol=solve_ivp(fevo,[t0,tend],[S0*kb/mh],method='RK45')

# plot our output!
plt.loglog(sol.t/yr,sol.y[0]/(kb/mh))
plt.xlim(t0/yr,tend/yr)
plt.title('entropy vs. time')
plt.ylabel(r'S [$k_{B}/m_{H}$]')
plt.xlabel('time [yr]')
plt.show()
```
