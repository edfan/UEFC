#! /usr/bin/env python3

import math
import numpy as np
from scipy.optimize import minimize_scalar, brute

import matplotlib.pyplot as plt

epsilon = 1e-5

class Plane():
    """
    Variable convention:
    Parameters are stored in the object using their names (listed below).
    A method to calculate a parameter has "c_" appended to the start of the
    parameter name. These methods set the result into the Plane object. These 
    methods do not take any arguments; they read from the Plane object.

    Ex. Calculating and setting W_wing:
    p = Plane(...various required parameters)
    p.c_W_wing() # Calculates W_wing and sets it in p.
    print(p.W_wing) # The value of W_wing.

    Parameter names:  
    A: cross-sectional area
    b: wing span
    c: average wing chord
    c_d0...c_d8: XFOIL-computed polars
    c_l: coefficient of airfoil lift (also c_l0 from XFOIL)
    c_r: root wing chord
    c_t: tip wing chord
    C_D: coefficient of drag
    C_Dfuse: coefficient of fuselage drag
    C_Di: coefficient of induced drag
    C_DP: coefficient of profile drag on wing
    CDA_0: fuselage drag
    delta: tip deflection
    delta_over_b: deflection/span ratio
    delta_over_b_max: maximum allowable deflection/span ratio
    e: span efficiency factor
    eps: airfoil camber ratio
    E_foam: foam Young's modulus (stiffness)
    g: acceleration due to gravity
    lamb: taper ratio (not lambda because of Python keyword)
    mu_air: dynamic viscosity of air
    N: load factor
    rho_air: air density
    rho_foam: foam density
    R: radius of circle
    Re: Reynolds number
    Re_ref: Reference Reynolds number
    tau: airfoil thickness ratio
    t_rev: time to complete one revolution
    T: thrust available
    T_max: maximum available thrust
    T_req: required thrust to match t_rev
    V: flight speed
    W: total weight
    W_max: maximum total weight (from T_max)
    W_supportable: maximum total weight (from T)
    W_pay: weight of payload
    W_pay_max: maximum payload weight supportable
    W_fuse: weight of fuselage
    W_wing: weight of wing
    Y: dihedral angle
    ---------
    AR: aspect ratio
    S: reference (wing) area
    C_L: coefficient of lift
    """
    
    def __init__(self, **kwargs):
        """
        Initialize plane with chosen variables.

        Args: Takes any number of keyword (named) arguments.
        Returns: New Plane object with given attributes set.
        """
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def check_vars(self, to_calc, required_vars):
        """
        Checks that a given calculation is currently possible, based on which
        variables have been defined in the Plane object. This function should
        be called prior to calculating any parameter.

        Args: to_calc: String; name of the parameter to be calculated (for 
                       debugging purposes).
              required_vars: Tuple; names of required variables in this 
                             calculation.
        Returns: True if calculation is valid; raises AttributeError otherwise.
        """

        all_vars_valid = True

        for r_v in required_vars:
            if r_v not in self.__dict__.keys():
                print("Cannot calculate", to_calc, "- missing", r_v)
                all_vars_valid = False

        if not all_vars_valid:
            raise AttributeError

        return all_vars_valid # True

    def c_A(self):
        required_vars = ['c', 'tau']
        self.check_vars('A', required_vars)
        
        self.A = 0.6 * (self.c)**2 * self.tau

    def c_AR(self):
        required_vars = ['b', 'c']
        self.check_vars('AR', required_vars)

        self.AR = self.b / self.c

    def c_b(self):
        required_vars = ['AR', 'S']
        self.check_vars('b', required_vars)

        self.b = (self.AR * self.S)**0.5
        
    def c_c(self):
        required_vars = ['S', 'AR']
        self.check_vars('c', required_vars)
        
        self.c = (self.S / self.AR)**0.5

    def c_c_r(self):
        required_vars = ['AR', 'b', 'lamb']
        self.check_vars('c_r', required_vars)

        self.c_r = 2 * self.b / (self.AR * (self.lamb + 1)) 
        
    def c_C_D(self):
        required_vars = ['S', 'C_DP', 'C_Di']
        self.check_vars('C_D', required_vars)

        self.CDA_0 = 0.002 + 0.002 * self.S / 0.228
        
        self.C_D = self.CDA_0 / self.S + self.C_DP + self.C_Di
            
    def c_C_DP(self):
        required_vars = ['c_l', 'Re', 'Re_ref', 'tau']
        self.check_vars('C_DP', required_vars)

        c_d0 = 0.02 * (1 + self.tau ** 2)
        c_d1 = -0.005 / (1 + 6.0 * self.tau)
        c_d2 = 0.16 / (1 + 60.0 * self.tau)
        c_d8 = 1.0
        c_l0 = 1.25 - 3.0 * self.tau
        
        lift_diff = self.c_l - c_l0
        self.C_DP = (c_d0 + c_d1 * lift_diff + c_d2 * lift_diff**2 + c_d8 * lift_diff**8) * (self.Re / self.Re_ref)**(-0.75)
        
    def c_C_Di(self):
        required_vars = ['C_L', 'AR', 'e']
        self.check_vars('C_Di', required_vars)

        self.C_Di = (self.C_L)**2 / (math.pi * self.AR * self.e)
        
    def c_c_l(self):
        required_vars = ['C_L']
        self.check_vars('c_l', required_vars)

        self.c_l = self.C_L

    def c_delta_over_b(self):
        required_vars = ['N', 'W_fuse', 'W_pay', 'E_foam', 'tau', 'eps', 'lamb', 'AR', 'S']
        self.check_vars('delta_over_b', required_vars)

        self.delta_over_b = 0.018 * self.N * \
            (self.W_fuse + self.W_pay) / (self.E_foam * self.tau * ((self.tau)**2 + 0.7*(self.eps)**2)) * \
            (1.0 + self.lamb)**3 * (1.0 + 2.0 * self.lamb) * (self.AR)**3 / self.S

    def c_e(self):
        required_vars = ['e0', 'Y', 'C_L', 'AR', 'b', 'R', 'N']
        self.check_vars('e', required_vars)

        r_bar = self.b / (2 * self.R * self.N)
        beta = self.C_L * (1 + 4.0 / self.AR) * r_bar / (self.Y * 2 * math.pi)

        self.e = self.e0 * (1 - 0.5 * r_bar**2) * (np.cos(beta))**2

    def c_eps(self):
        required_vars = ['tau']
        self.check_vars('tau', required_vars)

        self.eps = 0.1 - 0.5 * self.tau

    def c_lamb(self):
        required_vars = ['c_t', 'c_r']
        self.check_vars('lamb', required_vars)

        self.lamb = self.c_t / self.c_r

    def c_N(self, ignore_int=False):
        required_vars = ['W_fuse', 'W_wing', 'W_pay', 'rho_air', 'g', 'R', 'S', 'C_L']
        self.check_vars('N', required_vars)

        intermediate = (1 - ((self.W_fuse + self.W_wing + self.W_pay) / \
            (0.5 * self.rho_air * self.g * self.R * self.S * self.C_L))**2)
        
        if ignore_int:
            intermediate = abs(intermediate)
        else:
            if intermediate < 0:
                raise ValueError # Imaginary result produced otherwise.

        self.N = intermediate**-0.5

    def c_Re(self):
        required_vars = ['rho_air', 'V', 'c', 'mu_air']
        self.check_vars('Re', required_vars)

        self.Re = self.rho_air * self.V * self.c / self.mu_air

    def c_S(self):
        required_vars = ['b', 'c']
        self.check_vars('S', required_vars)

        self.S = self.b * self.c

    def c_t_rev(self):
        required_vars = ['R', 'V']
        self.check_vars('t_rev', required_vars)

        self.t_rev = 2 * math.pi * self.R / self.V
        
    def c_T(self):
        required_vars = ['N', 'W', 'C_D', 'C_L']
        self.check_vars('T', required_vars)

        self.T = self.N * self.W * self.C_D / self.C_L
        
    def c_T_req(self):
        required_vars = ['rho_air', 'V', 'S', 'C_D']
        self.check_vars('t_rev', required_vars)

        self.T_req = 0.5 * self.rho_air * (self.V)**2 * self.S * self.C_D
        
    def c_T_max(self):
        required_vars = ['V']
        self.check_vars('T_max', required_vars)

        self.T_max = 1.0 - 0.08 * self.V - 0.0028 * self.V**2

    def c_V(self, ignore_int=False):
        required_vars = ['g', 'R', 'N']
        self.check_vars('V', required_vars)

        intermediate = (self.N)**2 - 1

        if ignore_int:
            intermediate = abs(intermediate)
        else:
            if intermediate < 0:
                raise ValueError # Imaginary result produced otherwise.

        self.V = (self.g * self.R * (intermediate)**0.5)**0.5

    def c_V_max(self, ignore_int=False):
        required_vars = ['T']
        self.check_vars('V_max', required_vars)

        intermediate = 11 - 7 * self.T

        if ignore_int:
            intermediate = abs(intermediate)
            self.V_max = (50.0 / 7) * (intermediate**0.5 - 2)
        else:
            if intermediate < 0:
                raise ValueError # Imaginary result produced otherwise.

            self.V_max = max((50.0 / 7) * (intermediate**0.5 - 2),
                             (-50.0 / 7) * (intermediate**0.5 + 2))

    def c_W(self):
        required_vars = ['W_wing', 'W_fuse', 'W_pay']
        self.check_vars('W', required_vars)

        self.W = self.W_wing + self.W_fuse + self.W_pay

    def c_W_fuse(self):
        required_vars = ['b', 'g', 'S']
        self.check_vars('W_fuse', required_vars)

        self.W_fuse = (.145 + .06 * self.b / 1.52 + .045 * self.S / .228) * self.g

    def c_W_max(self):
        required_vars = ['T_max', 'C_D', 'C_L']
        self.check_vars('W_max', required_vars)

        self.W_max = self.T_max * self.C_L / self.C_D

    def c_W_supportable(self):
        required_vars = ['T', 'C_D', 'C_L']
        self.check_vars('W_supportable', required_vars)

        self.W_supportable = self.T * self.C_L / self.C_D

    def c_W_pay_max(self):
        required_vars = ['W_max', 'W_wing', 'W_fuse']
        self.check_vars('W_pay_max', required_vars)

        self.W_pay_max = self.W_max - self.W_wing - self.W_fuse

    def c_W_wing(self):
        required_vars = ['rho_foam', 'g', 'tau', 'b', 'c_r', 'lamb']
        self.check_vars('W_wing', required_vars)

        lmo = self.lamb - 1.0

        self.W_wing = 1.2 * self.b * self.tau * self.rho_foam * self.g * (self.c_r)**2 * \
                      (1.0/6 * lmo**2 + 0.5 * lmo + 0.5)
                                                                          
        

# Represent Plane Vanilla parameters as a dictionary, which can then be turned
# into a Plane object.
plane_vanilla_params = {
    'c_r': 0.2,
    'c_t': 0.1,
    'e0': 0.99,
    'E_foam': 19.0e6,
    'g': 9.81,
    'mu_air': 1.81e-5,
    'rho_air': 1.225,
    'rho_foam': 33.0,
    'R': 12.5,
    'Re_ref': 100000,
    'Y': 0.125664,
}

# Calculating W_pay_max for Plane Vanilla
p = Plane(**plane_vanilla_params)
p.c_lamb()

tau = 0
lamb = 0

def optimized_combined(args, delta_over_b_max, print_result=False, ignore_db=False):
    # args:
    # 0: S
    # 1: AR
    # 2: C_L (loaded)
    # 3: C_L (empty)
    # 4: W_pay

    global tau, lamb

    p = Plane(**plane_vanilla_params)
    p.lamb = lamb

    try:
        p.S = args[0]
        p.AR = args[1]
        p.C_L = args[2]
        p.W_pay = args[4]
        p.tau = tau
        p.c_b()
        p.c_c()
        p.c_A()
        p.c_W_fuse()
        p.c_W_wing()
        p.c_W()
        p.c_N(True)
        p.c_e()
        p.c_V(True)
        p.c_eps()
        p.c_c_r()
        p.c_c_l()
        p.c_C_Di()
        
        for _ in range(5):
            p.c_Re()

            if not ignore_db and (p.Re < 0 or p.Re > 200000):
                raise ValueError
            
            p.c_C_DP()
            p.c_C_D()
            p.c_T()
            p.c_T_req()
            p.c_V_max()
            p.V = p.V_max

            if not ignore_db and p.V < 0:
                raise ValueError
        
 
        p.c_t_rev()
        t_loaded = p.t_rev
        p.c_delta_over_b()

        p.c_T_req()
        p.c_T_max()
        p.c_W_max()

        if print_result:
            print(p.__dict__)

        if not ignore_db and p.delta_over_b > delta_over_b_max + epsilon:
            raise ValueError
        
        if not ignore_db and p.T_req > p.T_max + epsilon:
            raise ValueError

        if not ignore_db and p.W > p.W_max + epsilon:
            raise ValueError

        if print_result:
            print('For {}:'.format(delta_over_b_max))
            print('t_rev_min', p.t_rev)
            print('C_D', p.C_D)
            print('d/b', p.delta_over_b)
            print('V', p.V)
            print('T', p.T)
            print('T_req', p.T_req)
            print('T_max', p.T_max)
            print('N', p.N)
            print('\n')

        p.W_pay = 0
        p.C_L = args[3]
        p.c_W()
        p.c_N(True)
        p.c_e()
        p.c_V(True)
        p.c_C_Di()
        p.c_c_l()

        for _ in range(5):
            p.c_Re()

            if not ignore_db and (p.Re < 0 or p.Re > 200000):
                raise ValueError

            p.c_C_DP()
            p.c_C_D()
            p.c_T()
            p.c_T_req()
            p.c_V_max(True)
            p.V = p.V_max

            if not ignore_db and p.V < 0:
                raise ValueError
        
        p.c_t_rev()
        t_empty = p.t_rev
        p.c_delta_over_b()
        p.c_T_req()
        p.c_T_max()
        p.c_W_max()

        if print_result:
            print(p.__dict__)

        if not ignore_db and p.t_rev < 0:
            raise ValueError

        if not ignore_db and p.delta_over_b > delta_over_b_max + epsilon:
            raise ValueError
        
        if not ignore_db and p.T_req > p.T_max + epsilon:
            raise ValueError

        if not ignore_db and p.W > p.W_max + epsilon:
            raise ValueError

        result = args[4] / (t_loaded + t_empty)

        if print_result:
            print('For {}:'.format(delta_over_b_max))
            print('t_rev_min', p.t_rev)
            print('C_D', p.C_D)
            print('d/b', p.delta_over_b)
            print('V', p.V)
            print('T', p.T)
            print('T_req', p.T_req)
            print('T_max', p.T_max)
            print('N', p.N)
            print('\n')

            print('Final result:', result)
            
        return -1.0 * result
    
    except ValueError:
        return 10000

ranges = (slice(0.405, 0.44, 0.002), slice(13.3, 13.55, 0.005),
          slice(1.171, 1.172, 0.0002), slice(0.54, 0.5404, 0.0001),
          slice(4.18, 4.20, .002))

tau = 0.12
lamb = 0.5
#tmp = brute(optimized_combined, ranges, args=(0.15,), finish=None)
tmp = [0.411, 13.355, 1.171, .54, 4.196]
print('Combined (constrained for d/b <= 0.15):', tmp)
print(optimized_combined(tmp, 0.15, True))



# Graph S vs AR optimization

def simple_combined(AR, S):
    return optimized_combined((S, AR, tmp[2], tmp[3], tmp[4]), 0.15, False, True)

def simple_delta_over_b(AR, S):
    global tau
    
    p = Plane(**plane_vanilla_params)
    p.lamb = lamb

    p.S = S
    p.AR = AR
    p.C_L = tmp[2]
    p.W_pay = tmp[4]
    p.tau = tau
    p.c_b()
    p.c_c()
    p.c_A()
    p.c_W_fuse()
    p.c_W_wing()
    p.c_W()
    p.c_N(True)
    p.c_e()
    p.c_V(True)
    p.c_eps()
    p.c_c_r()
    p.c_c_l()
    p.c_C_Di()
    
    for _ in range(5):
        p.c_Re()        
        p.c_C_DP()
        p.c_C_D()
        p.c_T()
        p.c_T_req()
        p.c_V_max(True)
        p.V = p.V_max
              
 
    p.c_t_rev()
    t_loaded = p.t_rev
    p.c_delta_over_b()

    return p.delta_over_b

x = np.arange(10, 20, 0.1)
y = np.arange(0.3, 0.45, 0.001)
X,Y = np.meshgrid(x, y)
Z = []
for b in y:
    for a in x:
        Z.append(-1 * simple_combined(a, b))
Z = np.array(Z)
Z = Z.reshape((len(y), len(x)))
Z2 = simple_delta_over_b(X, Y)

plt.xlabel(r'AR')
plt.ylabel(r'S ($m^2$)')
plt.title('Combined optimization ($C_{L_{full}} = 1.171, C_{L_{empty}} = 0.54$)')

plt.plot(tmp[1], tmp[0], 'b.')

C1 = plt.contour(X, Y, Z, levels=[.12, .121, .122, .123, .124, .125], colors = '0.7')
plt.clabel(C1, inline=1, fontsize=10, fmt='%1.3f')
C2 = plt.contour(X, Y, Z2, inline=1, fontsize=10, levels=[0.15], colors=['r'])

"""
plt.annotate(r"$t_{rev}$ = 6.67 s", xy=(5.14, 0.0499), xytext=(4.5, 0.055),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.37 s", xy=(6.26, 0.0477), xytext=(5.6, 0.0545),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.22 s", xy=(7.04, 0.0467), xytext=(6.2, 0.056),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$T_{max}$ = 0.7 N", xy=(5.4, 0.0493), xytext=(5.1, 0.0445),
             arrowprops=dict(arrowstyle="->"))
"""

plt.show(block=False)

input("Continue...")
plt.close()

# C_L graphs

def simple_combined(C1, C2):
    return optimized_combined((tmp[0], tmp[1], C1, C2, tmp[4]), 0.15, False, True)

def simple_delta_over_b(C1, C2):
    global tau, lamb
    
    p = Plane(**plane_vanilla_params)
    p.lamb = lamb

    p.S = tmp[0]
    p.AR = tmp[1]
    p.C_L = C1
    p.W_pay = tmp[4]
    p.tau = tau
    p.c_b()
    p.c_c()
    p.c_A()
    p.c_W_fuse()
    p.c_W_wing()
    p.c_W()
    p.c_N(True)
    p.c_e()
    p.c_V(True)
    p.c_eps()
    p.c_c_r()
    p.c_c_l()
    p.c_C_Di()
    
    for _ in range(5):
        p.c_Re()
        p.c_C_DP()
        p.c_C_D()
        p.c_T()
        p.c_T_req()
        p.c_V_max(True)
        p.V = p.V_max
              
 
    p.c_t_rev()
    t_loaded = p.t_rev
    p.c_delta_over_b()

    return p.delta_over_b

def simple_T_diff(C1, C2):
    global tau, lamb

    p = Plane(**plane_vanilla_params)
    p.lamb = lamb

    p.S = tmp[0]
    p.AR = tmp[1]
    p.C_L = C2
    p.W_pay = 0
    p.tau = tau
    p.c_b()
    p.c_c()
    p.c_A()
    p.c_W_fuse()
    p.c_W_wing()
    p.c_W()
    p.c_N(True)
    p.c_e()
    p.c_V(True)
    p.c_eps()
    p.c_c_r()
    p.c_c_l()
    p.c_C_Di()

    for _ in range(5):
        p.c_Re()

        p.c_C_DP()
        p.c_C_D()
        p.c_T()
        p.c_T_req()
        p.c_V_max(True)
        p.V = p.V_max
        
    p.c_t_rev()
    t_empty = p.t_rev
    p.c_delta_over_b()
    p.c_T_req()
    p.c_T_max()

    return p.T_max - p.T_req
    
x = np.arange(1.05, 1.25, 0.01)
y = np.arange(0.5, 0.6, 0.01)
X,Y = np.meshgrid(x, y)
Z = []
for b in y:
    for a in x:
        Z.append(-1 * simple_combined(a, b))
Z = np.array(Z)
Z = Z.reshape((len(y), len(x)))
#Z2 = simple_delta_over_b(X, Y)
Z3 = simple_T_diff(X, Y)

plt.xlabel(r'C_L (loaded)')
plt.ylabel(r'C_L (unloaded)')
plt.title('Combined optimization ($S = 0.411 m^2, AR = 13.355$)')

plt.plot(tmp[2], tmp[3], 'b.')

print(Z3[:100])

C1 = plt.contour(X, Y, Z, levels=[0.12, 0.121, 0.122, 0.123, 0.124, 0.125], colors = '0.7')
plt.clabel(C1, inline=1, fontsize=10, fmt='%1.3f')
#C2 = plt.contour(X, Y, Z2, inline=1, fontsize=10, levels=[0.15], colors=['r'])
C3 = plt.contour(X, Y, Z3, inline=1, fontsize=10, levels=[0], colors=['g'])


"""
plt.annotate(r"$t_{rev}$ = 6.67 s", xy=(5.14, 0.0499), xytext=(4.5, 0.055),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.37 s", xy=(6.26, 0.0477), xytext=(5.6, 0.0545),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.22 s", xy=(7.04, 0.0467), xytext=(6.2, 0.056),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$T_{max}$ = 0.7 N", xy=(5.4, 0.0493), xytext=(5.1, 0.0445),
             arrowprops=dict(arrowstyle="->"))
"""

plt.show(block=False)

input("Continue...")
plt.close()
