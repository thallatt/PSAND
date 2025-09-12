# PSAND: Planetary Structure ANd Dynamics
A python method to couple planetary structure and dynamics, as published in Hallatt & Millholland (2025).

minimal working example:

import numpy as np

fTdm,fL,fR=np.load('fTm_mc=10_z=0_1au.npy',allow_pickle=True).item(),np.load('fL_mc=10_z=0_1au.npy',allow_pickle=True).item(),np.load('fR_mc=10_z=0_1au.npy',allow_pickle=True).item()

def dSdtm(S,Lextra_,M,Tirr): return (-fL(S,Tirr)+Lextra_)/fTm(S,Tirr)
