"""
Bouncing ball simulation

Estimate the ball with n points around its circumference.
The force between each point is Hookean.
The force between each point and the ground should go to infinity as the ball approaches the ground.
The pressure on each point is directed away from the average position of the points
It has magnitude inversely proportional to the distance from the center of the ball squared
There is also gravity on each point.
"""

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

#constants
n = 100 # number of points
g = 1e1 # gravity
pr = 6e2 # force on each point from pressure
el = 3e5 # force on each point from elasticity
normal = 1e3 # normal force
radius = 0.5 # of ball
initialHeight = 8 # of ball center

spacing = 2*np.pi*radius / n # spacing of particles

stretch = 2 # initial stretch in the ball

# animation constants
frames = 1000
calcperframe = 20 # calculations of state between each frame
timestep = 1 # between calcs
animationspeed = 30

xlim = (-6, 6)
ylim = (-2, 10)

#initial shape
ball = np.zeros((n, 2))
for i in range(0, n):
    ball[i, 0] = radius * np.cos(i / n * 2*np.pi) * stretch
    ball[i, 1] = radius * np.sin(i / n * 2*np.pi) * stretch + initialHeight

# velocity
vel = np.zeros((n, 2))

# force of gravity on the points
def gravityForce():
    Fgx = np.zeros((n, 1))
    Fgy = -np.ones((n, 1)) * g
    Fg = np.concatenate((Fgx, Fgy), axis = 1)
    return Fg

# elastic force between points
def elasticForce():
    Fe = np.zeros((n, 2))
    for i in range(0, n):
        iplus = (i + 1) % n
        iminus = (i - 1) % n
        dplus = np.linalg.norm(ball[iplus, :] - ball[i, :]) # distance to next point
        dminus = np.linalg.norm(ball[iminus, :] - ball[i, :])
        fplus = (ball[iplus, :] - ball[i, :])*(1 - spacing / dplus) # force from next point
        fminus = (ball[iminus, :] - ball[i, :])*(1 - spacing / dminus)
        Fe[i, :] = el * (fplus + fminus)
    return Fe

# force between the ground and the ball
def normalForce():
    Nx = np.zeros(n)
    Ny = (ball[:, 1] < 0)*normal
    N = np.transpose(np.array([Nx, Ny]))
    return N

# force of pressure from the inflated ball
def pressure():
    middlex = np.average(ball[:, 0])
    middley = np.average(ball[:, 1])
    Fp = np.zeros((n, 2))
    for i in range(0, n):
        mag = ((middlex - ball[i, 0])**2 + (middley - ball[i, 1])**2)**(1/2)
        Fp[i, 0] = pr*(ball[i, 0] - middlex) / mag**3
        Fp[i, 1] = pr*(ball[i, 1] - middley) / mag**3
    return Fp

# add gravitational, elastic, normal force, and pressure
def totalForce():
    Fg = gravityForce()
    Fe = elasticForce()
    N = normalForce()
    Fp = pressure()
    return Fg + Fe + N + Fp

# generate data
balldata = np.zeros((frames, n, 2))
balldata[0, :] = ball

for i in range(1, frames):
    for k in range(1, calcperframe):
        F = totalForce()
        vel = vel + F * timestep / 1000
        # dampen velocity of particles relative to each other
        vel = (2*vel + np.roll(vel, 1, 0) + np.roll(vel, -1, 0)) / 4
        # update ball position
        ball = ball + vel * timestep / 1000
    # record ball position    
    balldata[i, :] = ball
    
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=xlim, ylim=ylim)
drawball, = ax.plot([], [], 'r', lw=1)
ground, = ax.plot(xlim, [0, 0], 'g', lw=1)

def update_pic(i):
    drawball.set_data(balldata[i, :, 0], balldata[i, :, 1])
    return drawball,
    
    
anim = animation.FuncAnimation(fig, update_pic, frames = frames, interval = timestep*animationspeed, blit = True)
anim.save('bouncingball.gif')
    
    


