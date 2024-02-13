#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:33:57 2024

@author: Julien Hacot-SLonosky
@collab: Guilherme Caumo, Ben Cheung
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting timestep and total number of timesteps
dt = 5
Nsteps = 50

# Vortex rings initial conditions
y_v = np.array([-50, 50, -50, 50], dtype="f")  # y-positions of the 4 vortices
x_v = np.array([-50, -50, 50, 50], dtype="f")  # x-positions of the 4 vortices
k_v = np.array([40, -40, 40, -40])  # line vortex constants for the 4 vortices

# Set up the plot
plt.ion()
fig, ax = plt.subplots(1, 1)
p, = ax.plot(x_v, y_v, 'k+', markersize=10)  # You can adjust marker size and type

# Simulation grid
ngrid = 200
Y, X = np.mgrid[-ngrid:ngrid:360j, -ngrid:ngrid:360j]  # Adjust resolution as needed
vel_x = np.zeros(np.shape(Y))  # x-velocity
vel_y = np.zeros(np.shape(Y))  # y-velocity

# Masking radius
r_mask = 5

# Defining a function to calculate velocity field
def calculate_velocity(vortex_x, vortex_y, k_v, X, Y, r_mask=0):
    r = np.sqrt((X - vortex_x)**2 + (Y - vortex_y)**2)  # find the radius b/w
                                                        # the vortices
    r_masked = np.where(r < r_mask, np.nan, r)
    
    theta = np.arctan2(Y - vortex_y, X - vortex_x) # compute angle between the vortices
    
    vel_x = k_v * np.sin(theta) / r_masked  # compute x velocity
    vel_y = -k_v * np.cos(theta) / r_masked  # compute y velocity
    
    return vel_x, vel_y

# Initial velocity field calculation
for i in range(len(x_v)):
    vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
    vel_x += vx
    vel_y += vy

ax.set_xlim([-ngrid, ngrid])
ax.set_ylim([-ngrid, ngrid])
ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])
fig.canvas.draw()

# Evolution
count = 0
while count < Nsteps:
    vel_x = np.zeros(np.shape(Y))  # reset the x velocities
    vel_y = np.zeros(np.shape(Y))  # reset the y velocities
    
    for i in range(len(x_v)):
        # Compute the velocites
        vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
        # update the velocities
        vel_x += vx
        vel_y += vy
    
    # Update positions of vortices
    for i in range(len(x_v)):
        x_v[i] += dt * np.nanmean(vel_x[(Y >= y_v[i] - r_mask) & (Y <= y_v[i] + r_mask) &
                                        (X >= x_v[i] - r_mask) & (X <= x_v[i] + r_mask)])
        
        y_v[i] += dt * np.nanmean(vel_y[(Y >= y_v[i] - r_mask) & (Y <= y_v[i] + r_mask) &
                                        (X >= x_v[i] - r_mask) & (X <= x_v[i] + r_mask)])
    
    p.set_xdata(x_v)
    p.set_ydata(y_v)

    # Clear and update streamlines
    for coll in ax.collections:
        coll.remove()

    for patch in ax.patches:
        patch.remove()
    
    ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])
    fig.canvas.draw()
    plt.pause(0.001)
    
    count += 1
