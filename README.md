# PSAND: Planetary Structure ANd Dynamics
A python method to couple planetary structure and dynamics, as published in [Hallatt & Millholland (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250922923H/abstract).

If you use PSAND, please cite [Hallatt & Millholland (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250922923H/abstract).

The structure grids cover temperatures 288-2884 K, core masses {10, 20} Earth masses, and total masses [10.1,300] Earth masses. We are currently improving the high metallicity structure models, so we only include solar composition models in the current version of PSAND.

Data files for interpolation take up 110 MB.

Please reach out to me if you would like to collaborate, have questions, or if you would like custom structure models to be created. I can be reached at thallatt@mit.edu.

## A minimal working example

This example loads our interpolation functions assuming solar composition gas, 250 Earth mass planet with a 10 Earth mass core, and an equilibrium temperature of 289 K. We integrate equation 5 from Hallatt & Millholland (2025). We then plot the entropy, luminosity, and radius as a function of time.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# import supermongo python plotting library (https://github.com/AstroJacobLi/smplotlib)
import smplotlib

# define constants (cgs units)
yr = 31560000.0
kB = 1.3807e-16
mH = 1.6726e-24
sigSB = 5.67e-5
RE = 637800000.0

# planet mass, core mass string, initial entropy, equilibrium temperature (constant for this example),
# metallicity ('002': Z=0.02), extra luminosity (constant for this example)
M = 250      # ME
mc = '10'    # ME
S0 = 10      # kB/mH
Teq = 289   # 288 K is the current minimum
z = '002'
extraL = 0. # erg/s

# output name
outname = str(M)+'ME_mc'+mc+'_S0'+str(S0).replace('.','')+'_Teq'+str(int(Teq))+'_z'+z\
+'_Lx'+str(extraL).replace('+','')

# load interpolation functions for planet structure
fTdm = np.load('data/fTm_mc'+mc+'_z'+z+'.npy',allow_pickle=True).item()
fL = np.load('data/fL_mc'+mc+'_z'+z+'.npy',allow_pickle=True).item()
fR = np.load('data/fR_mc'+mc+'_z'+z+'.npy',allow_pickle=True).item()

# define extra heating rate (erg/s) as a function of time (s). for this example, we set it to zero.
def Lextra(t):
  return extraL

# evolve planet entropy by "stepping through the adiabats" (equation 5 of Hallatt & Millholland (2025)).
def dSdtm(S,Lextra_,M,Teq):
  return (-fL(S,M,Teq) + Lextra_)/fTdm(S,M,Teq)

# differential equation to be solved with scipy
def fevo(t,y,M,Teq):
  S = y[0]
  Lextra_ = Lextra(t)
  dsdt = dSdtm(S,Lextra_,M,Teq)
  return [dsdt]

# set time range
t0, tend = 1e6, 1e10

# integrate thermal evolution!
# implicit methods (e.g. BDF) are more stable and faster than explicit.
sol = solve_ivp(fevo, [t0*yr,tend*yr], [S0*kB/mH], method = 'BDF', args = (M,Teq))

# save output from our integration
Sout = sol.y[0]
Rout = fR(Sout,M,Teq)
Lout = fL(Sout,M,Teq)

# plot output
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10, 4))

ax1.set_ylabel(r'$S$ [$k_{\rm B}/m_{\rm H}$]')
ax1.set_xlabel('time [yr]')
ax1.semilogx(sol.t/yr, Sout/(kB/mH))
ax1.set_xticks([1e6,1e7,1e8,1e9,1e10])
ax1.set_ylim(6,11)
ax1.set_yticks(np.arange(6,12))

ax2.set_ylabel(r'log $L$ [erg/s]')
ax2.set_xlabel('time [yr]')
ax2.loglog(sol.t/yr, np.log10(Lout))
ax2.set_xticks([1e6,1e7,1e8,1e9,1e10])
ax2.set_ylim(24,30)
ax2.set_yticks(np.arange(24,31))

ax3.set_ylabel(r'R [R$_\oplus$]')
ax3.set_xlabel('time [yr]')
ax3.semilogx(sol.t/yr, Rout)
ax3.set_xticks([1e6,1e7,1e8,1e9,1e10])

fig.tight_layout()
fig.savefig('SLR_'+outname+'.pdf',bbox_inches='tight')

# output data file
outfile = 'planetevolution_'+outname+'.csv'
outdat = pd.DataFrame(np.array([sol.t/yr, Sout/(kB/mH), Rout, Lout])).T
outdat.columns = ['t/yr', 'S/(kB/mH)', 'R/RE', 'L/(erg/s)']
outdat.to_csv(outfile,index=False)
```
![example_evolution](SLR_250ME_Lx0.png "example_evolution")
