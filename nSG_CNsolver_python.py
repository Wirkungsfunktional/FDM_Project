#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
import matplotlib.pyplot as plt
from multiprocessing import Pool
# parameters


l = 1

def residual(P, u0, h, dt):
    d2x = zeros_like(P) + 0.j
    d2u = zeros_like(P) + 0.j
    h2 = h*h

    d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) /h2
    d2x[0]    = (P[1]    - 2*P[0]    + P[-1])  /h2
    d2x[-1]   = (P[0]    - 2*P[-1]   + P[-2])  /h2
    
    d2u[1:-1] = (u0[2:]   - 2*u0[1:-1] + u0[:-2]) /h2
    d2u[0]    = (u0[1]    - 2*u0[0]    + u0[-1])  /h2
    d2u[-1]   = (u0[0]    - 2*u0[-1]   + u0[-2])  /h2

    return 2.j * (P - u0)/dt + d2u + d2x - l*(np.abs(u0)**2*u0 + np.abs(P)**2*P)


# solve


def u0(x):
    return np.exp(1.j*x)

def ana_u(x, t):
    return np.exp(1.j*x + 1.j*(-l - 1)*t)

    



def nSG_solve(N, xa, xe, ta, te, dt, func_u0):
    h = (xe - xa) / (N+1)
    x = np.arange(xa+h, xe+h/2, h)
    u0 = func_u0(x)
    t = [ta]
    while (ta < te):
        u0 = newton_krylov(lambda a: residual(a, u0, h, dt), u0)
        ta += dt
        t.append(ta)
    
    return u0, x, np.array(t)
    

def test_plot():
    u, x, t = nSG_solve(400, -np.pi, np.pi, 0, 0.01, 0.00001, u0)
    
    plt.plot(x, np.abs(u)**2 )
    plt.plot(x, np.abs( ana_u(x, t[-1]) )**2 )
    
    plt.show()


def nSG_l1_space_error(n):
    u, x, t = nSG_solve(n, -np.pi, np.pi, 0, 1, 0.01, u0)
    return np.sum(np.abs( u - ana_u(x, t[-1]) ))
    
def nSG_l1_time_error(t):
    u, x, T = nSG_solve(400, -np.pi, np.pi, 0, 1, t, u0)
    return np.sum(np.abs( u - ana_u(x, T[-1]) ))

def nSG_space_conv_plot():
    N = [50, 100, 150, 200, 250, 300, 350, 400]
    p = Pool()
    err = p.map(nSG_l1_space_error, N)
    print err
    
    h = 2*np.pi/(np.array(N)+1)
    plt.loglog(h, h*np.array(err) )
    plt.plot(h, h**2)
    plt.show()

def nSG_time_conv_plot():
    T = np.array([0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
    p = Pool()
    err = p.map(nSG_l1_time_error, T)
    print err
    
    plt.loglog(T, T*np.array(err) )
    plt.plot(T, T**2)
    plt.show()


def main(args):
    #test_plot()
    nSG_space_conv_plot()
    nSG_time_conv_plot()
    
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))



