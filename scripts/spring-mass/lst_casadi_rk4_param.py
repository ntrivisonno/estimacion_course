# LST Least-Squares-Technique for the Dynamic System - estimating physics parameters

"""
@author: ntrivisonno
"""

import casadi as cs
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

y_sim = np.loadtxt("./mediciones/y_sim.txt")
#y_sim = np.loadtxt("./mediciones/x_sim_noise.txt")
u_sim = np.loadtxt("./mediciones/u_sim.txt")
x_sim = np.loadtxt("./mediciones//x_sim.txt")

Nx = 2
Nw = Nx
Nu = 1
Ny = 1
Nv = Ny
Np = 2

Ts = 0.1  # Sampling time, should match the sampling of the generated data.

Nsim = y_sim.shape[0]  # Total batch length
Nsim -= 1


# Optimizer variables
opt_var = ctools.struct_symSX([ctools.entry('x', shape = (Nx, 1), repeat = Nsim+1),
                               ctools.entry('v', shape = (Nv, 1), repeat = Nsim+1),
                               ctools.entry('w', shape = (Nw, 1), repeat = Nsim),
                               ctools.entry('p', shape = (Np, 1))])

# Optimizer parameters
opt_par = ctools.struct_symSX([ctools.entry('y', shape = (Ny, 1), repeat = Nsim+1),
                               ctools.entry('u', shape = (Nu, 1), repeat = Nsim)])


P_mhe_x0 = cs.DM.eye(Nx) # Arrival Cost weighting matrix [x0bar]
P_mhe_p0 = cs.DM.eye(Np)
#P_mhe_p0 = cs.DM.eye(Np) * 2E-4 # Arrival Cost weighting matrix [p0bar]
#P_mhe_p0 = cs.DM([[2e-2, 0.0],[0.0, 2.8e-3]])
Q_mhe = cs.DM.eye(Nw) # Process Weighting matrix
R_mhe = cs.DM.eye(Nv) # Measurements Weighting matrix

states_constraints = []
states_constraints_lb = []
states_constraints_ub = []

measurements_constraints = []
measurements_constraints_lb = []
measurements_constraints_ub = []

optimization_variables_lb = []
optimization_variables_ub = []

# Definition of the objetive function for optimization
J = 0.0
#J += cs.mtimes([(opt_var['x',0]-opt_par['x0bar']).T, P_mhe_x0, (opt_var['x',0]-opt_par['x0bar'])])
#J += cs.mtimes([(opt_var['p']-opt_par['p0bar']).T, P_mhe_p0, (opt_var['p']-opt_par['p0bar'])])

for _k in range(Nsim-1):
    J += cs.mtimes([opt_var['v', _k].T, R_mhe, opt_var['v', _k]])
    J += cs.mtimes([opt_var['w', _k].T, Q_mhe, opt_var['w', _k]])

# Next iteration out of the for-loop, _k = N-1
J += cs.mtimes([opt_var['v', Nsim-1].T, R_mhe, opt_var['v', Nsim-1]])

# Symbolic variables definition for the RK4 function
x = cs.SX.sym('x',Nx)
v = cs.SX.sym('v',Nv)
w = cs.SX.sym('w',Nw)
u = cs.SX.sym('u',Nu)
y = cs.SX.sym('y',Ny)
p = cs.SX.sym('p', Np)

# Spring-Mass system
m = 20.0  # Mass

# Dynamic System - ODE for estimating physical states and parameters
f_rhs = cs.vertcat(x[1], - p[0]/m * x[0] - p[1]/m * x[1] + 1/m * u)
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

for _k in range(Nsim-1):
    states_constraints += [opt_var['x', _k+1] - F_RK4(opt_var['x', _k], opt_par['u', _k], opt_var['p']) - opt_var['w', _k]]
    states_constraints_lb += [cs.DM.zeros(Nx)]
    states_constraints_ub += [cs.DM.zeros(Nx)]

    measurements_constraints += [opt_par['y', _k] - h(opt_var['x', _k]) - opt_var['v', _k]]
    measurements_constraints_lb += [cs.DM.zeros(Ny)]
    measurements_constraints_ub += [cs.DM.zeros(Ny)]

    
# Next iteration out of the for-loop, _k = N-1
measurements_constraints += [opt_par['y', Nsim-1] - h(opt_var['x', Nsim-1]) - opt_var['v', Nsim-1]]
states_constraints_lb += [cs.DM.zeros(Nx)]
states_constraints_ub += [cs.DM.zeros(Nx)]

nlp_constraints = states_constraints + measurements_constraints
nlp_constraints = cs.vertcat(*nlp_constraints)

#-------------------------------------------
#                 NLP
#-------------------------------------------
# Create an NLP solver
prob = {'f': J,
        'x': opt_var,
        'g': nlp_constraints,
        'p': opt_par}
solver = cs.nlpsol('solver', 'ipopt', prob)

curr_par = opt_par(0) # Generates current_parameters identical as struct opt_par, filled all with zeros
initialization_state = opt_var(0) # Generates states identical as struct opt_var, filled all with zeros
# inicialization state personalized, x0 = [1,1], p0 = [1,1]
for i in range(Nsim+1):
    initialization_state["x",i] = cs.DM([1,1])
initialization_state["p"] = cs.DM([0.5,1])

# Current parameters
curr_par['y', lambda __x: cs.horzcat(*__x)] = y_sim[:,np.newaxis]
curr_par['u', lambda __x: cs.horzcat(*__x)] = u_sim[:-1,np.newaxis]

optimization_variables_lb = opt_var(-cs.inf)
optimization_variables_lb['p'] = cs.DM([0.0, 0.0])
optimization_variables_ub = opt_var(cs.inf)
# optimization_variables_ub['p'] = cs.DM([10.0, 10.0])

#-------------------------------------------
# Solve the NLP
sol = solver(x0=initialization_state,
             lbx=optimization_variables_lb, ubx=optimization_variables_ub,
             lbg=0, ubg=0, p=curr_par)

x_estimated = cs.DM.zeros(Nx, Nsim)
p_estimated = cs.DM.zeros(Np, Nsim)

curr_sol = opt_var(sol['x'])  # Generates an struct call current_solution similar as struct opt_var, and also assign the solution of the mhe windows
x_estimated = curr_sol['x', lambda __x: cs.horzcat(*__x)]
p_estimated = curr_sol['p']

# calc error posicion
p0_hat = np.mean(p_estimated[0,int(Nsim-Nsim/3):-1])
p1_hat = np.mean(p_estimated[1,int(Nsim-Nsim/3):-1])
dstd_p0 = np.std(p_estimated[0,int(Nsim-Nsim/3):-1])
dstd_p1 = np.std(p_estimated[1,int(Nsim-Nsim/3):-1])
 
err_p0_hat = ((p0_hat-2.0)/2.0)*100
err_p1_hat = ((p1_hat-4.0)/4.0)*100

err = x_estimated[0,:] - y_sim[:,np.newaxis].T
err = err.T
err_abs = cs.fabs(err)

print('#--------------------------------------------')
print("N_batch: {}".format(Nsim))
print("p_estimated: {}".format(p_estimated))
print("p[0]_hat: {:.4f}, sigma:+-{:.4f} -> p_orig: 2.0 -> err rel: {:.4f}%".format(p0_hat, dstd_p0, err_p0_hat))
print("p[1]_hat: {:.4f}, sigma:+-{:.4f} -> p_orig: 4.0 -> err rel: {:.4f}%".format(p1_hat, dstd_p1, err_p1_hat))


print('#--------------------------------------------')
print('\n FIN, OK!')


# Numerical simulation of spring-mass system (x_sim)
#plt.figure(1)
#plt.plot(x_sim[0,:].toarray().flatten(), label='x0_sim')
#plt.plot(x_sim[1,:].toarray().flatten(), label='x1_sim')
# Noised measurement
plt.figure(2)
plt.plot(y_sim[:], label='$\overline{y}$')
plt.plot(x_estimated[0,:-1].toarray().flatten(), label='$\hat{x}_0$')
plt.plot(x_estimated[1,:-1].toarray().flatten(), label='$\hat{x}_1$')
plt.title('Spring-Mass Numerical Simulator')
plt.legend()
plt.grid()

plt.figure(3)
plt.subplot(211)
plt.plot(y_sim, marker="+",label="$\overline{y}$")
plt.plot(x_estimated[0,:-1].toarray().flatten(), label='$\hat{x}_0$')
plt.title('Spring-Mass system')
plt.legend()
plt.grid()
plt.subplot(212)
plt.plot(cs.fabs(err[:-1]), label="$\epsilon_{\hat{x_0}}$")
plt.title('Error')
plt.ylabel('$|\epsilon|$')
plt.legend()
plt.grid()

plt.figure(4)
plt.subplot(211)
plt.plot(p_estimated[0,:].toarray().flatten(), label='$\hat{p_{[0]}}$')
plt.title('Estimation p[0] - k')
plt.legend()
plt.grid()
plt.subplot(212)
plt.plot(p_estimated[1,:].toarray().flatten(), label='$\hat{p_{[1]}}$')
plt.title('Estimation p[1] - c')
plt.legend()
plt.grid()

plt.show()