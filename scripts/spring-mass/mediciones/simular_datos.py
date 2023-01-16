import casadi as cs
import numpy as np

import matplotlib.pyplot as plt

# Define the size of the variables and paramters we are going to use
Nx = 2
Nw = Nx
Nu = 1
Ny = 1
Nv = Ny
Np = 2

# Sampling time
Ts = 0.1

# Total batch length
Nsim = 600

# Initial state
x0 = cs.DM([-2.0, 0.0])

# Symbolic variables definition for the RK4 function
x = cs.SX.sym('x', Nx)
v = cs.SX.sym('v', Nv)
w = cs.SX.sym('w', Nw)
u = cs.SX.sym('u', Nu)
y = cs.SX.sym('y', Ny)
p = cs.SX.sym('p', Np)

# Spring-Mass system
c = 4.0  # Damping constant
k = 2.0  # Stiffness of the spring
m = 20.0  # Mass

# p = cs.vertcat(k, c)

# Dynamics System - ODE for estimating physical parameters
f_rhs = cs.vertcat(x[1], - p[0] / m * x[0] - p[1] / m * x[1] + 1 / m * u[0])


f = cs.Function('f', [x, u, p], [f_rhs])

# Observation equation
h_rhs = [x[0]]
h_rhs = cs.vertcat(*h_rhs)

# Dynamics discretization
# Fixed step RK4 integrator
M = 4  # RK4 steps per interval
for j in range(1, M):
    k1 = f(x, u, p)
    k2 = f(x + Ts / 2 * k1, u, p)
    k3 = f(x + Ts / 2 * k2, u, p)
    k4 = f(x + Ts * k3, u, p)
    x_rk4 = x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Generation of Function from the ODE system
F_RK4 = cs.Function('F', [x, u, p], [x_rk4])
h = cs.Function('h', [x], [h_rhs])


usim = np.ones((Nsim, Nu)) * 5.0
xsim = np.zeros((Nsim + 1, Nx))
ysim = np.zeros((Nsim, Ny))
psim = np.array([k, c])
xsim[0] = np.ravel(x0)
for i in range(Nsim):
    ysim[i] = h(xsim[i])
    xsim[i+1] = np.array(F_RK4(xsim[i], usim[i], psim)).ravel()


plt.figure()
plt.plot(xsim[:, 0], label="${x_0}$")
plt.plot(xsim[:, 1], label="${x_1}$")
plt.grid()
plt.legend()
plt.show()

np.savetxt('y_sim.txt', ysim)
np.savetxt('u_sim.txt', usim)
np.savetxt('x_sim.txt', xsim)
