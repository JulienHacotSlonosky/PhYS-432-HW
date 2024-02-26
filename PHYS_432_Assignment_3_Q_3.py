# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:49:27 2024

@author: Julien, Eve J. Lee
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up the grid and the advection and diffusion parameters
Ngrid = 50
Nsteps = 5000
dt = 1
dx = 1

v = -0.16/60 # length of a monster can divided by the time it took to go past it.
alpha = v*dt/2/dx
D1 = 1e3

beta1 = D1*dt/dx**2

g = -9.8
nu = 100 #Pa s 

x = np.arange(Ngrid)*dx

# Initial Conditions
f1 = np.copy(x)* 0.0  # Assume no initial flow


# set up plot
plt.ion()
fig, axes = plt.subplots(1,1)
axes.set_title('lava flow')


# Plotting initial state in the background for reference
axes.plot(x, f1, 'k-')


# These plotting objects will be updated
plt1, = axes.plot(x, f1, 'ro')
#plt2, = axes[1].plot(x, f2, 'ro')

axes.set_ylim([0,Ngrid])


fig.canvas.draw()

count = 0
#%%

geff = g * np.sin(10 * (np.pi/180)) # compute the gravtiational term
# Evolution
while count < Nsteps:
    ## Calculating diffusion first
    # Setting the matrices for diffuion operator
    A1 = np.eye(Ngrid) * (1.0 + 2.0 * beta1) + np.eye(Ngrid, k=1) * -beta1 + np.eye(Ngrid, k=1) * -beta1
    
    ## Boundary conditions for no-stress
    # This ensures that f in the last cell stays fixed at all times under diffusion
    A1[-1][-1] = 1 + beta1

    #No-slip boundary conditions for the first cell
    A1[0][0] = 1.0
    A1[0][1] = 0

    
    ## Calculate advection
    # Lax-Griedrichs
    u = f1
    u[1:] = u[1:] + geff*dt # Adding the gravitational term
    # Solving for the next timestep
    f1 = np.linalg.solve(A1, u)

    # update the plot
    plt1.set_ydata(f1)

    
    fig.canvas.draw()
    plt.pause(0.001)
    count += 1
