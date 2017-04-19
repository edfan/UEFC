from gpkit import Variable, Model
import gpkit
import numpy as np

# Constants
c_l0 = Variable("c_l0", 0.8, "-", "lift coefficient 0")
CDA_0 = Variable("CDA_0", 0.004, "m^2", "fuselage drag area")
e = Variable("e", 0.95, "-", "efficiency factor")
eps = Variable("\\epsilon", 0.03, "-", "airfoil camber ratio")
E_foam = Variable("E_{foam}", 19.3e6, "Pa", "Young's modulus of foam")
g = Variable("g", 9.81, "m/s^2", "acceleration due to gravity")
N = Variable("N", 1.0, "-", "load factor")
rho_air = Variable("\\rho_{air}", 1.225, "kg/m^3", "density of air")
rho_foam = Variable("\\rho_{foam}", 32.0, "kg/m^3", "density of foam")
R = Variable("R", 12.5, "m", "turning radius")
tau = Variable("\\tau", 0.11, "-", "airfoil thickness ratio")
T_max = Variable("T_{max}", 0.7, "N", "maximum thrust")
W_fuse = Variable("W_{fuse}", 2.7, "N", "fuselage weight")

C_L = Variable("C_L", "-", "lift coefficient")
S = Variable("S", "m^2", "planform area")
AR = Variable("AR", "-", "aspect ratio")

b = Variable("b", "m", "wing span")
c_r = Variable("c_r", "m", "root chord")
C_DP = Variable("C_DP", "-", "wing drag coefficient")
C_Di = Variable("C_Di", "-", "induced drag coefficient")
C_D = Variable("C_D", "-", "drag coefficient")
W_max = Variable("W_max", "N", "total supportable weight")
W_wing = Variable("W_wing", "N", "wing weight")
W_pay = Variable("W_pay", "N", "payload weight")
delta_over_b = Variable("delta_over_b", "-", "deflection/span ratio")

W_int = Variable("W_int", "N", "W_fuse + W_pay")
bend_int = Variable("bend_int", "-", "tau^2 + eps^2")

# Constraint
with gpkit.SignomialsEnabled():
	constraints = [b == (AR * S)**0.5,
				   c_r >= 2 * (AR * S)**0.5 / (AR * 1.5),
                   C_DP >= 0.02 - 0.004 * (C_L - c_l0) + \
                   	       0.02 * (C_L - c_l0)**2 + (C_L - c_l0)**8,
                   C_Di >= C_L**2 / (np.pi * AR * e),
                   C_D >= 0.004 + C_DP + C_Di,
                   W_max <= T_max * C_L / C_D,
                   W_wing >= 22.8 / 24 * b * tau * rho_foam * g * c_r**2,
                   W_pay + W_wing + W_fuse <= W_max,
                   W_int >= W_fuse + W_pay,
                   bend_int >= tau**2 + eps**2,
                   delta_over_b >= 0.0045 * N * W_int / \
                                   (E_foam * tau * bend_int) * \
                                   AR**3 / S,
                   delta_over_b <= 0.15]

# Objective (to minimize)
objective = 1.0 / W_pay

# Formulate the Model
m = Model(objective, constraints)

# Solve the Model
sol = m.localsolve(verbosity=0, iteration_limit=100)

# print solution table
print sol.table()
