# PSAND: Planetary Structure ANd Dynamics
A python method to couple planetary structure and dynamics, as published in Hallatt & Millholland (2025).

## A minimal working example

```
import numpy as np
import consts
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Tirr=900.
M=250.

fTdm=np.load('fTm_mc=10_z=0_1au.npy',allow_pickle=True).item()
fL=np.load('fL_mc=10_z=0_1au.npy',allow_pickle=True).item()
fR=np.load('fR_mc=10_z=0_1au.npy',allow_pickle=True).item()

def dSdtm(S,Lextra_,M,Tirr): return (-fL(S,Tirr)+Lextra_)/fTm(S,Tirr)

def fevo(t,y):
  S=y[0]
  dsdt=dSdtm(S,Lextra_,M,Tirr)
  return [dsdt]

t0,tend=0.,1e10*consts.yr
S0=8.
sol=solve_ivp(fevo,[t0,tend],[S0*consts.k/consts.mH],method='RK45')

plt.loglog(sol.t/consts.yr,sol.y[0]/(consts.k/consts.mH))
plt.title('entropy vs. time')
plt.ylabel(r'S [$k_{B}/m_{H}$]')
plt.xlabel('time [yr]')
plt.show()
```
