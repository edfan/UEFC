import numpy as np
import gpkit
from gpkit import Variable, Model

# Constants
# b = Variable("b", 1.52, "m", "wing span")
c_d0 = Variable("c_d0", 0.02, "-", "drag coefficient 0")
c_d1 = Variable("c_d1", -0.004, "-", "drag coefficient 1")
c_d2 = Variable("c_d2", 0.02, "-", "drag coefficient 2")
c_d8 = Variable("c_d8", 1.0, "-", "drag coefficient 8")
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

constraints = []


with gpkit.SignomialsEnabled():
    # Free variables
    C_L = Variable("C_L", "-", "lift coefficient")
    S = Variable("S", "m^2", "planform area")
    AR = Variable("AR", "-", "aspect ratio")
    #lamb = Variable("lamb", "-", "taper ratio")
    p1 = Variable("p1", "-", "2*lambda + 1")
    p2 = Variable("p2", "-", "lambda + 1")
    constraints += [2*p1 >= 1 + p2]
    p3 = Variable("p3", "-", "C_L - c_l0")
    constraints += [C_L >= 0.8 + p3]
    p4 = Variable("p4", "-", "lambda - 1")
    constraints += [p2 >= p4 + 2]

    # Dependent variables

    b = Variable("b", "m", "wing span")
    constraints += [b >= (AR * S)**0.5]

    c_r = Variable("c_r", "m", "root chord")
    constraints += [c_r >= 2 * (AR * S)**0.5 / (AR * p2)]

    C_DP = Variable("C_DP", "-", "wing drag coefficient")
    constraints += [C_DP >= c_d0 + c_d1 * p3 + c_d2 * p3**2 + \
                    c_d8 * p3**8]

    C_Di = Variable("C_Di", "-", "induced drag coefficient")
    constraints += [C_Di >= C_L**2 / (np.pi * AR * e)]

    C_D = Variable("C_D", "-", "drag coefficient")
    constraints += [C_D >= CDA_0 / S + C_DP + C_Di]

    W_max = Variable("W_max", "N", "total supportable weight")
    constraints += [W_max <= T_max * C_L / C_D]

    W_wing = Variable("W_wing", "N", "wing weight")
    constraints += [W_wing >= 1.2 * b * tau * rho_foam * g * c_r**2 * \
                    (1.0/6 * p4**2 + 0.5 * p4 + 0.5)]

    W_pay = Variable("W_pay", "N", "payload weight")
    constraints += [W_pay + W_wing + W_fuse <= W_max]

    p5 = Variable("p5", "N", "fuselage + payload weight")
    constraints += [p5 >= W_fuse + W_pay]

    p6 = Variable("p6", "-", "tau^2 + eps^2")
    constraints += [p6 >= tau**2 + eps**2]

    delta_over_b = Variable("delta_over_b", "-", "deflection/span ratio")
    constraints += [delta_over_b >= 0.018 * N * p5 / \
                    (E_foam * tau * p6) * p2**3 * \
                    p1 * AR**3 / S]

    # Constraints
    constraints += [delta_over_b <= 0.15]

# Results
m = Model(1.0/W_pay, constraints)
sol = m.solve(verbosity=0)
print(sol.table())

