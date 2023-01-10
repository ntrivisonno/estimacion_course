# MHE with an ODE for the Dynamic System
# Funciona OK!
"""

@author: ntrivisonno
"""


import casadi as cs
import casadi.tools as ctools
import numpy as np
#from OOP.src import utils
import matplotlib.pyplot as plt


y_sim = np.loadtxt("/home/zeeburg/Documents/CIMEC/Cursos/estimacion_course/scripts/spring-mass/mediciones/y_medido_20220929.txt")
u_sim = np.loadtxt("/home/zeeburg/Documents/CIMEC/Cursos/estimacion_course/scripts/spring-mass/mediciones/u_20220929.txt")

Nx = 2
Nw = Nx
Nu = 1
Ny = 1
Nv = Ny
N = 5 # Horizon Windows
Ts = 0.1 # Sampling time
Nsim = 600 # Total batch length
x0 = cs.DM([-2.0, 0.0]) # Initial state

opt_var = ctools.struct_symSX([ctools.entry('x', shape = (Nx, 1), repeat = N),
                               ctools.entry('v', shape = (Nv, 1), repeat = N),
                               ctools.entry('w', shape = (Nw, 1), repeat = N-1)])
opt_par = ctools.struct_symSX([ctools.entry('x0bar', shape = (Nx, 1)),
                               ctools.entry('y', shape = (Ny, 1), repeat = N),
                               ctools.entry('u', shape = (Nu, 1), repeat = N-1)])

P_mhe = cs.DM.eye(Nx) # Arrival Cost weighting matrix
Q_mhe = cs.DM.eye(Nw) # Process Weighting matrix
R_mhe = cs.DM.eye(Nv) # Measurements Weighting matrix

states_constraints = []
states_constraints_lb = []
states_constraints_ub = []

measurements_constraints = []
measurements_constraints_lb = []
measurements_constraints_ub = []

J = 0.0
# Definition of the objetive function for optimization
J += cs.mtimes([(opt_var['x',0]-opt_par['x0bar']).T, P_mhe, (opt_var['x',0]-opt_par['x0bar'])])

for _k in range(N-1):
    J += cs.mtimes([opt_var["v", _k].T, R_mhe, opt_var['v', _k]])
    J += cs.mtimes([opt_var["w", _k].T, Q_mhe, opt_var['w', _k]])

# Next iteration out of the for-loop, _k = N-1
J += cs.mtimes([opt_var["v", N-1].T, R_mhe, opt_var['v', N-1]])

# Symbolic variables definition for the optimization problem
x = cs.SX.sym('x',Nx)
v = cs.SX.sym('v',Nv)
w = cs.SX.sym('w',Nw)
u = cs.SX.sym('u',Nu)
y = cs.SX.sym('y',Ny)

# Spring-Mass system
c = 4.0  # Damping constant
k = 2.0  # Stiffness of the spring
m = 20.0 # Mass

A = np.array([[0.0, 1.0], [-k / m, -c / m]])
B = np.array([[0.0], [1.0 / m]])
C = np.array([1, 0])
D = 0

#------------------------------------------
# Dynamics System - ODE
f_ode = [cs.mtimes(A, x) + cs.mtimes(B, u)]

f_ode = cs.vertcat(*f_ode)
# Observation equation
h_rhs = [x[0]]

h_rhs = cs.vertcat(*h_rhs)
#------------------------------------------

f = cs.Function('f',[x,u], [f_ode])

# Dynamics discretization
# Fixed step RK4 integrator
M = 4 # RK4 steps per interval
for j in range(1, M):
    k1 = f(x, u)
    k2 = f(x + Ts / 2 * k1, u)
    k3 = f(x + Ts / 2 * k2, u)
    k4 = f(x + Ts * k3, u)
    x_rk4 = x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Generation of Function from the ODE system
F_RK4 = cs.Function('F', [x, u], [x_rk4])
h = cs.Function('h', [x], [h_rhs]) 

for _k in range(N-1):
    states_constraints += [opt_var['x', _k+1] - F_RK4(opt_var['x', _k], opt_par['u', _k]) - opt_var['w', _k]]
    states_constraints_lb += [cs.DM.zeros(Nx)]
    states_constraints_ub += [cs.DM.zeros(Nx)]

    measurements_constraints += [opt_par["y", _k] - h(opt_var['x', _k]) - opt_var['v', _k]]
    measurements_constraints_lb += [cs.DM.zeros(Ny)]
    measurements_constraints_ub += [cs.DM.zeros(Ny)]
    
# Next iteration out of the for-loop, _k = N-1
measurements_constraints += [opt_par["y", N-1] - h(opt_var['x', N-1]) - opt_var['v', N-1]]
measurements_constraints_lb += [cs.DM.zeros(Ny)]
measurements_constraints_ub += [cs.DM.zeros(Ny)]

nlp_constraints = states_constraints + measurements_constraints
#nlp_constraints = cs.vertcat(*nlp_constraints) #deconstructor of the list, inside the dictionary

#-------------------------------------------
# Create an NLP solver
prob = {'f': J,
        'x': opt_var,
        'g': cs.vertcat(*nlp_constraints),
        'p': opt_par}
solver = cs.nlpsol('solver', 'ipopt', prob)

curr_par = opt_par(0) # Generates current_parameters identical as struct opt_par, filled all with zeros
initialization_state = opt_var(0) # Generates states identical as struct opt_var, filled all with zeros

# Current parameters
curr_par['x0bar'] = x0
curr_par['y', lambda __x: cs.horzcat(*__x)] = y_sim[0:N]
curr_par['u', lambda __x: cs.horzcat(*__x)] = u_sim[0:N-1]

optimization_variables_lb = opt_var(-cs.inf)
optimization_variables_ub = opt_var(cs.inf)
# if want different bounds of each struct:
#optimization_variables_lb['x', lambda __x: cs.horzcat(*__x)] = np.array([[8, 8, 8, 8, 8], [8, 8, 8, 8, 8]])
#optimization_variables_lb['v', lambda __x: cs.horzcat(*__x)] = np.array([[3, 3, 3, 3, 3]])
#optimization_variables_lb['w', lambda __x: cs.horzcat(*__x)] = np.array([[9, 9, 9, 9], [9, 9, 9, 9]])

#-------------------------------------------
# Solve the NLP
sol = solver(x0=initialization_state,
             lbx = optimization_variables_lb, ubx = optimization_variables_ub, 
             lbg=0, ubg=0, p = curr_par)

x_estimated = cs.DM.zeros(Nx, Nsim)

curr_sol = opt_var(sol['x']) # Generates an struct call current_solution similar as struct opt_var, and also assign the solution of the mhe windows
x_estimated[:,:N] = curr_sol['x', lambda __x: cs.horzcat(*__x)]

# MHE windows, starts rolling
current_x0bar = x_estimated[:, 1]
for i in range(N,Nsim):
    curr_par['x0bar'] = current_x0bar
    curr_par['y', lambda __x: cs.horzcat(*__x)] = y_sim[i-N:i]
    curr_par['u', lambda __x: cs.horzcat(*__x)] = u_sim[i-N:i-1]

    sol = solver(x0 = sol['x'],
                 lbx = optimization_variables_lb, ubx = optimization_variables_ub, 
                 lbg=0, ubg=0, p = curr_par)
    
    curr_sol = opt_var(sol['x'])
    x_estimated[:,i] = curr_sol['x', N-1]
    
    current_x0bar = x_estimated[:, i-N+2]


# calc error posicion
err = x_estimated[0,:] - y_sim[:-1,np.newaxis].T
err = err.T

print('#--------------------------------------------')
print('\n FIN, OK!')

# plt.plot(x_rk4)

# Numerical simulation of spring-mass system (x_sim)
#plt.figure(1)
#plt.plot(x_sim[0,:].toarray().flatten(), label='x0_sim')
#plt.plot(x_sim[1,:].toarray().flatten(), label='x1_sim')
# Noised measurement
plt.figure(2)
plt.plot(y_sim[:], label='$\overline{y}$')
plt.plot(x_estimated[0,:].toarray().flatten(), label='$\hat{x}_0$')
plt.plot(x_estimated[1,:].toarray().flatten(), label='$\hat{x}_1$')
plt.title('Spring-Mass Numerical Simulator')
plt.legend()

plt.figure(3)
plt.subplot(211)
plt.plot(y_sim, marker="+",label="$\overline{y}$")
plt.plot(x_estimated[0,:].toarray().flatten(), label='$\hat{x}_0$')
plt.title('Spring-Mass system')
plt.legend()
plt.subplot(212)
plt.plot(err, label="$\epsilon_{\hat{x_0}}$")
plt.title('Error')
plt.legend()

plt.show()
