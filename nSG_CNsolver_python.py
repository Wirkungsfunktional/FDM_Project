#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
import matplotlib.pyplot as plt
# parameters

nx = 400
hx = 2*np.pi/(nx)
dt = 0.001
l = 0.5

def residual(P, u0):
    d2x = zeros_like(P) + 0.j
    d2u = zeros_like(P) + 0.j

    d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) /hx/hx
    d2x[0]    = (P[1]    - 2*P[0]    + P[-1])  /hx/hx
    d2x[-1]   = (P[0]    - 2*P[-1]   + P[-2])  /hx/hx
    
    d2u[1:-1] = (u0[2:]   - 2*u0[1:-1] + u0[:-2]) /hx/hx
    d2u[0]    = (u0[1]    - 2*u0[0]    + u0[-1])  /hx/hx
    d2u[-1]   = (u0[0]    - 2*u0[-1]   + u0[-2])  /hx/hx

    return 2.j * (P/dt - u0/dt) + d2u + d2x - l*np.abs(u0)**2*u0 - l*np.abs(P)**2*P

# solve
x = np.linspace(-np.pi, np.pi, nx)
u0 = np.exp(1.j*x)

for i in range(1, 100):
    u0 = newton_krylov(lambda x: residual(x, u0), u0)
    
plt.plot(x, np.abs(u0)**2 )
    
plt.plot(x, np.abs(np.exp(1.j*x + 1.j*(l - 1)*i*dt) )**2 )
plt.show()

# visualize



