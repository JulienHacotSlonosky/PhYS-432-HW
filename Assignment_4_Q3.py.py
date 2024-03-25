# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:02:13 2024

@author: Julien Hacot-Slonosky, Andrew Cumming, Maryn Askew
"""

import numpy as np
import matplotlib.pyplot as plt

def advect_step(q, u, dt, dx):
    
    #calculate fluxes
    J = np.zeros(n-1)
    
    for i in range(n-1):
        if u[i]>0.0:
            J[i] = q[i]*u[i]
        else:
            J[i] = q[i+1]*u[i]
    # Now do the update
    q[1:-1] = q[1:-1] - (dt/dx)*(J[1:] - J[:-1])
    
    q[0] = q[0] - (dt/dx)*J[0]
    q[-1] = q[-1] + (dt/dx)*J[-1]
    
    return q

nsteps = 10000 # number of steps
alpha = 0.1  # time step
gamma = 5/3  # adiabatic index

n = 1000
x = np.linspace(0,1,n)
dx = x[1]-x[0]
dt = alpha * dx
q1 = np.ones(n)  # I.C. rho = 1
q2 = np.zeros(n)  # I.C. u = 0
q3 = np.ones(n)  # I.C. rho*e_tot = 1

u = q2 / q1
e_tot = q3 / q1
e_kin = 0.5*(u**2)
P = (gamma - 1) * q1 * (e_tot - e_kin)  # Pressure from the Heidelburg lectures
cs2 = gamma * (P / q1) 

# ------------------- Initital Conditions ------------------------------------
AA = 50 # amplitude

# Gaussian density with amplitude AA
q3 = 1.0 + AA * np.exp(-(x-0.5)**2/0.0025)
#q2 = 1.0 * AA*np.exp(-(x-0.2)**2/0.004)
mach = (1/ np.sqrt(cs2)) * (q2/q1)
# -----------------------------------------------------------------------------

# Set up the plots
plt.ion()

plt.subplot(211)
x2, = plt.plot(x, q1,'bo',ms=1)
plt.xlim([0,1])
plt.ylim([-2,4])
plt.ylabel("Density")

plt.subplot(212)
x1, = plt.plot(x,q2,'ro',ms=1)
plt.xlim([0,1])
plt.ylim([-2,2])
plt.ylabel("Mach Number")
plt.xlabel('x')

plt.draw()

# now do the iterations
count = 0

while count < nsteps:
    
    # compute advection velocity at the cell and sim boundaries
    u = 0.5 * ((q2[:-1]/q1[:-1]) + (q2[1:]/q1[1:]))
    # advect density and momentum
    q1 = advect_step(q1,u,dt,dx)
    q2 = advect_step(q2,u,dt,dx)
    
    # compute pressure
    P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    # add the pressure source term
    q2[1:-1] = q2[1:-1] - dt * (P[2:]-P[:-2]) / (2.0 * dx)
    # Boundary Conditions for Reflective Boundaries
    q2[0] = q2[0] - dt * (P[1] - P[0])/(2*dx)
    q2[-1] = q2[-1] - dt * (P[-1]-P[-2])/(2*dx)
    
    # recompute the advection velocities
    u = 0.5 * ((q2[:-1]/q1[:-1]) + (q2[1:]/q1[1:]))
    
    # advect energy
    q3 = advect_step(q3,u,dt,dx)
    
    # recompute pressure
    P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    
    # add the energy source term
    u = q2/q1
    q3[1:-1] = q3[1:-1] - dt * (P[2:]*u[2:]-P[:-2]*u[:-2])/(2.0*dx)
    q3[0] = q3[0] - dt * (P[1]*u[1]-P[0]*u[0])/(2*dx)
    q3[-1] = q3[-1] - dt * (P[-1]*u[-1]-P[-2]*u[-2])/(2*dx)
    
    # Recompute pressure and the speed of sound
    P = P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    cs2 = (1/q1) * gamma * P
    
    # update Mach number
    mach = (1/np.sqrt(cs2))*(q2 / q1)
    # update the plot
    x1.set_ydata(mach)
    x2.set_ydata(q1)
    plt.draw()
    plt.pause(0.001)
    
    count += 1