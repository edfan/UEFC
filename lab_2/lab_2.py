#! /usr/bin/env python3

import math
import numpy as np
from scipy.optimize import minimize_scalar, brute

import matplotlib.pyplot as plt

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
    N: load factor
    rho_air: air density
    rho_foam: foam density
    R: radius of circle
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
        required_vars = ['c_t', 'c_r']
        self.check_vars('c', required_vars)
        
        self.c = (self.c_t + self.c_r) / 2

    def c_c_r(self):
        required_vars = ['AR', 'b', 'lamb']
        self.check_vars('c_r', required_vars)

        self.c_r = 2 * self.b / (self.AR * (self.lamb + 1)) 
        
    def c_C_D(self):
        required_vars = ['CDA_0', 'S', 'C_DP', 'C_Di']
        self.check_vars('C_D', required_vars)
        
        self.C_D = self.CDA_0 / self.S + self.C_DP + self.C_Di
            
    def c_C_DP(self):
        required_vars = ['c_d0', 'c_d1', 'c_d2', 'c_d8', 'c_l', 'c_l0']
        self.check_vars('C_DP', required_vars)
        
        lift_diff = self.c_l - self.c_l0
        self.C_DP = (self.c_d0 + self.c_d1 * lift_diff + self.c_d2 * lift_diff**2 + self.c_d8 * lift_diff**8) #/ self.c_l
        
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
            (self.W_fuse + self.W_pay) / (self.E_foam * self.tau * ((self.tau)**2 + (self.eps)**2)) * \
            (1.0 + self.lamb)**3 * (1.0 + 2.0 * self.lamb) * (self.AR)**3 / self.S

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

    def c_S(self):
        required_vars = ['b', 'c']
        self.check_vars('S', required_vars)

        self.S = self.b * self.c

    def c_t_rev(self):
        required_vars = ['R', 'V']
        self.check_vars('t_rev', required_vars)

        self.t_rev = 2 * math.pi * self.R / self.V

    def c_T_req(self):
        required_vars = ['rho_air', 'V', 'S', 'C_D']
        self.check_vars('t_rev', required_vars)

        self.T_req = 0.5 * self.rho_air * (self.V)**2 * self.S * self.C_D

    def c_T(self):
        required_vars = ['N', 'W', 'C_D', 'C_L']
        self.check_vars('T', required_vars)

        self.T = self.N * self.W * self.C_D / self.C_L

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

    def c_W(self):
        required_vars = ['W_wing', 'W_fuse', 'W_pay']
        self.check_vars('W', required_vars)

        self.W = self.W_wing + self.W_fuse + self.W_pay

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

        #self.W_wing = 0.2 * self.b * self.tau * self.rho_foam * \
        #              self.g / (self.c_r - self.c_t) * (self.c_r**3 - self.c_t**3)

        lmo = self.lamb - 1.0

        self.W_wing = 1.2 * self.b * self.tau * self.rho_foam * self.g * (self.c_r)**2 * \
                      (1.0/6 * lmo**2 + 0.5 * lmo + 0.5)

        #self.W_wing = 0.15 * self.tau * self.rho_foam * self.g * (self.AR)**1.5 * \
        #              (self.S)**-0.5 * (self.lamb + 1)**2 * \
        #              (1.0/3 * lmo**2 + lmo + 1)
                                                                            
        

# Represent Plane Vanilla parameters as a dictionary, which can then be turned
# into a Plane object.
plane_vanilla_params = {
    'b': 1.52,
    'c_d0': 0.02,
    'c_d1': -0.004,
    'c_d2': 0.02,
    'c_d8': 1.0,
    'c_l0': 0.8,
    'c_r': 0.2,
    'c_t': 0.1,
    'C_L': 0.8,
    'CDA_0': 0.004,
    'delta': 0.18,
    'e': 0.95,
    'eps': 0.03,
    'E_foam': 19.3e6,
    'g': 9.81,
    'N': 1, # maximum load factor
    'rho_air': 1.225,
    'rho_foam': 32.0,
    'R': 12.5,
    'tau': 0.11,
    'T_max': 0.7,
    'W_fuse': 2.7
}

# Calculating W_pay_max for Plane Vanilla
p = Plane(**plane_vanilla_params)
p.c_lamb()
p.c_c()
p.c_c_l()
p.c_A()
p.c_AR()
p.c_S()
p.c_C_DP()
p.c_C_Di()
p.c_C_D()
p.c_W_wing()
p.c_W_max()
p.c_W_pay_max()
p.W_pay = p.W_pay_max
p.c_delta_over_b()
                                    
print('W_pay_max (unconstrained):', p.W_pay_max)
print('delta/b (unconstrained):', p.delta_over_b)
print('C_D', p.C_D)

# Find W_pay_max for delta_over_b <= 0.1.

def constrained_W_pay(W_pay, delta_over_b_max, print_result=False):
    p.W_pay = W_pay
    p.c_delta_over_b()

    if p.delta_over_b > delta_over_b_max:
        return 0

    if print_result:
        print('For {}:'.format(delta_over_b_max))
        print('W_pay_max', p.W_pay)
        print('C_D', p.C_D)
        print('\n')
    
    return -1.0 * W_pay # Find maximum through minimum.

ranges = (slice(0, 5, 0.01),)

tmp = brute(constrained_W_pay, ranges, args=(0.1,), finish=None)
print('W_pay_max (constrained for d/b <= 0.1):', tmp)
constrained_W_pay(tmp, 0.1, True)

# Plots for W_pay_max

def W_pay_from_T(T):
    p.T = T
    p.c_W_supportable()
    return p.W_supportable - p.W_wing - p.W_fuse

def delta_over_b_from_T(T):
    p.T = T
    p.c_W_supportable()
    p.W_pay = p.W_supportable - p.W_wing - p.W_fuse
    p.c_delta_over_b()
    return p.delta_over_b

x = np.linspace(0, .9, 100)
y = W_pay_from_T(x)
z = delta_over_b_from_T(x)

fig, ax1 = plt.subplots()
ax1.plot(x, y, color='b')
ax1.set_xlabel('Thrust (N)')
ax1.set_ylabel(r'$W_{pay}$ (N)')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x, z, color='g')
ax2.set_ylabel(r'$\frac{\delta}{b}$')
ax2.set_autoscaley_on(False)
ax2.set_ylim([-0.02, 0.4])
ax2.tick_params('y', colors='g')

plt.axvline(x=0.7, color='r')
plt.axvline(x=0.42, color='m', linestyle='dashed') 
plt.axhline(y=0.1, color='r')

plt.title('Payload supportable for given thrust')
plt.annotate(r"(T=0.7, $W_{pay}$=6.1)", xy=(0.7, 0.3), xytext=(0.43, 0.35),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"(T=0.42, $\frac{\delta}{b}$=0.1)", xy=(0.42, 0.1), xytext=(0.45, 0.03),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"(T=0.42, $W_{pay}$=2.27)", xy=(0.42, 0.18), xytext=(0.07, 0.20),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$T_{max}$ = 0.7 N", xy=(0.7, 0.15), xytext=(0.76, 0.12),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.1", xy=(0.8, 0.1), xytext=(0.76, 0.03),
             arrowprops=dict(arrowstyle="->"))

fig.tight_layout()
plt.show(block=False)

input("Continue...")
plt.close(fig)

# Find t_rev_min for delta_over_b = 0.1, W_pay = 0.

p.W_pay = 0
p.c_N()
p.c_V()
p.c_C_D()
p.c_t_rev()
p.c_T_req()
p.c_delta_over_b()

print('t_rev_min:', p.t_rev)
print('C_D:', p.C_D)
print('d/b:', p.delta_over_b)
print('V:', p.V)
print('T:', p.T_req)
print('N:', p.N)

# Plots for t_rev_min

def t_rev_from_T(T):
    p.T = T
    return 2 * math.pi * p.R * \
        (p.rho_air * p.S * p.C_D / (2 * p.T))**0.5

x = np.linspace(0.01, .9, 100)
y = t_rev_from_T(x)
z = delta_over_b_from_T(x)

fig, ax1 = plt.subplots()
ax1.plot(x, y, color='b')
ax1.set_xlabel('Thrust (N)')
ax1.set_ylabel(r'$t_{rev}$ (N)')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x, z, color='g')
ax2.set_ylabel(r'$\frac{\delta}{b}$')
ax2.set_autoscaley_on(False)
ax2.set_ylim([-0.02, 0.4])
ax2.tick_params('y', colors='g')

plt.axvline(x=0.7, color='r')
plt.axhline(y=0.1, color='r')
plt.axvline(x=0.252, color='m', linestyle='dashed') 
p.W_pay = 0
p.c_delta_over_b()
plt.axhline(y=p.delta_over_b, color='g')

plt.title(r'Minimum $t_{rev}$ for given thrust')
plt.annotate(r"$T_{max}$ = 0.7 N", xy=(0.7, 0.3), xytext=(0.76, 0.35),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.1", xy=(0.67, 0.1), xytext=(0.73, 0.124),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{actual}$ = 0.056",
             xy=(0.67, p.delta_over_b), xytext=(0.73, 0.08), arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}(T)_{implied}$", xy=(0.56, 0.14), xytext=(0.45, 0.20),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"(T=0.26, $t_{rev}$=13.93)", xy=(0.252, 0.036), xytext=(0.3, -0.01),
             arrowprops=dict(arrowstyle="->"))

fig.tight_layout()
plt.show(block=False)

input("Continue...")
plt.close(fig)

# Now, optimize W_pay_max

def optimized_W_pay(args, delta_over_b_max, print_result=False, ignore_db=False):
    p.N = 1 # back to maximum load factor
    p.S = args[0]
    p.AR = args[1]
    p.C_L = args[2]
    p.c_b()
    p.c_c_r()
    p.c_c_l()
    p.c_C_DP()
    p.c_C_Di()
    p.c_C_D()
    p.c_W_wing()
    p.c_W_max()

    p.W_pay = p.W_max - p.W_wing - p.W_fuse
    p.c_delta_over_b()
    if not ignore_db and p.delta_over_b > delta_over_b_max:
        return 0

    if print_result:
        print('For {}:'.format(delta_over_b_max))
        print('W_pay_max', p.W_pay)
        print('C_D', p.C_D)
        print('\n')
    
    return -1.0 * p.W_pay # Find maximum through minimum.

ranges = (slice(0.01, 1, 0.01), slice(1, 25, .1), slice(0.01, 1, 0.01))

#tmp = brute(optimized_W_pay, ranges, args=(0.1,), finish=None)
#print('W_pay_max (constrained for d/b <= 0.1):', tmp)
#optimized_W_pay(tmp, 0.1, True)

#tmp = brute(optimized_W_pay, ranges, args=(0.05,), finish=None)
#print('W_pay_max (constrained for d/b <= 0.05):', tmp)
#optimized_W_pay(tmp, 0.05, True)

#tmp = brute(optimized_W_pay, ranges, args=(0.15,), finish=None)
#print('W_pay_max (constrained for d/b <= 0.15):', tmp)
#optimized_W_pay(tmp, 0.15, True)

# Graph W_pay_max optimization

def simple_W_pay(AR, S):
    return -1.0 * optimized_W_pay((S, AR, 0.92), 10000, False, True)

def simple_delta_over_b(AR, S):
    p.N = 1
    p.S = S
    p.AR = AR
    p.C_L = 0.92
    p.c_b()
    p.c_c_r()
    p.c_c_l()
    p.c_C_DP()
    p.c_C_Di()
    p.c_C_D()
    p.c_W_wing()
    p.c_W_max()

    p.W_pay = p.W_max - p.W_wing - p.W_fuse
    p.c_delta_over_b()

    return p.delta_over_b

x = np.arange(0.1, 25, 0.1)
y = np.arange(0.01, 1, 0.01)
X,Y = np.meshgrid(x, y)
Z = simple_W_pay(X, Y)
Z2 = simple_delta_over_b(X, Y)

plt.xlabel(r'AR')
plt.ylabel(r'S ($m^2$)')
plt.title('Optimization of $W_{pay}$ ($C_{L} = 0.92$)')

plt.plot(8.9, 0.51, 'k.')
plt.plot(11, 0.56, 'k.')
plt.plot(12.9, 0.65, 'k.')

C1 = plt.contour(X, Y, Z, levels=[3, 4, 5, 6, 7, 8], colors = '0.7')
plt.clabel(C1, inline=1, fontsize=10, fmt='%1.1f')
C2 = plt.contour(X, Y, Z2, inline=1, fontsize=10, levels=[0.05, 0.1, 0.15], colors=['r', 'g', 'b'])
CB2 = plt.colorbar(C2, shrink=0.8, extend='both', pad = 0.05, label=r"$\frac{\delta}{b}_{max}$")

plt.annotate(r"$W_{pay}$ = 5.52 N", xy=(8.9, 0.51), xytext=(1, 0.75),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$W_{pay}$ = 6.80 N", xy=(11, 0.56), xytext=(17, 0.5),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$W_{pay}$ = 7.61 N", xy=(12.9, 0.65), xytext=(17, 0.6),
             arrowprops=dict(arrowstyle="->"))

plt.show(block=False)

input("Continue...")
plt.close()

# Graph 1D W_pay with d/b points

plt.axhline(y=0, color='0.7', linestyle='dashed')
plt.plot(5.52, 0, 'r.')
plt.plot(6.80, 0, 'g.')
plt.plot(7.61, 0, 'b.')

plt.xlabel(r'$W_{pay}$')
plt.title(r'$W_{pay}$ optimization results')

plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.05", xy=(5.52, 0), xytext=(5.21, 0.01),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.1", xy=(6.80, 0), xytext=(6.49, 0.01),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.15", xy=(7.61, 0), xytext=(7.3, 0.01),
             arrowprops=dict(arrowstyle="->"))

plt.gca().axes.get_yaxis().set_visible(False)
plt.xlim([4.75, 8])
plt.show(block=False)

input("Continue...")
plt.close()
    
# Optimize t_rev
def optimized_t_rev(args, delta_over_b_max, print_result=False, ignore_db=False):
    try:
        p.W_pay = 0
        p.S = args[0]
        p.AR = args[1]
        p.C_L = args[2]
        p.c_b()
        p.c_c_r()
        p.c_c_l()
        p.c_C_DP()
        p.c_C_Di()
        p.c_C_D()
        p.c_W_wing()
        p.c_W()
        p.c_N()
        p.c_V()
        p.c_t_rev()
        p.c_delta_over_b()
        p.c_T_req()
        
        if not ignore_db and p.delta_over_b > delta_over_b_max:
            raise ValueError

        if not ignore_db and p.T_req > p.T_max:
            raise ValueError

        if print_result:
            print('For {}:'.format(delta_over_b_max))
            print('t_rev_min', p.t_rev)
            print('C_D', p.C_D)
            print('d/b', p.delta_over_b)
            print('V', p.V)
            print('T', p.T_req)
            print('N', p.N)
            print('\n')
        
        return p.t_rev
    
    except ValueError:
        return 10000 # Arbitrarily large value; won't be the min.

ranges = (slice(0.045, 0.055, 0.0001), slice(4, 8, .01), slice(0.01, 1.01, 0.01))
    
#tmp = brute(optimized_t_rev, ranges, args=(0.1,), finish=None)
#print('t_rev_min (constrained for d/b <= 0.1):', tmp)
#optimized_t_rev(tmp, 0.1, True)

#tmp = brute(optimized_t_rev, ranges, args=(0.05,), finish=None)
#print('t_rev_min (constrained for d/b <= 0.05):', tmp)
#optimized_t_rev(tmp, 0.05, True)

#tmp = brute(optimized_t_rev, ranges, args=(0.15,), finish=None)
#print('t_rev_min (constrained for d/b <= 0.15):', tmp)
#optimized_t_rev(tmp, 0.15, True)

# Graph t_rev optimization

def simple_t_rev(AR, S):
    return optimized_t_rev((S, AR, 1.0), 10000, False, True)

def simple_delta_over_b(AR, S):
    p.W_pay = 0
    p.S = S
    p.AR = AR
    p.C_L = 1.0
    p.c_b()
    p.c_c_r()
    p.c_c_l()
    p.c_C_DP()
    p.c_C_Di()
    p.c_C_D()
    p.c_W_wing()
    p.c_W()
    p.c_N(True)
    p.c_V(True)
    p.c_t_rev()
    p.c_delta_over_b()
    
    return p.delta_over_b

def simple_T_req(AR, S):
    p.W_pay = 0
    p.S = S
    p.AR = AR
    p.C_L = 1.0
    p.c_b()
    p.c_c_r()
    p.c_c_l()
    p.c_C_DP()
    p.c_C_Di()
    p.c_C_D()
    p.c_W_wing()
    p.c_W()
    p.c_N(True)
    p.c_V(True)
    p.c_t_rev()
    p.c_delta_over_b()
    p.c_T_req()
    
    return p.T_req

x = np.arange(4, 7.5, 0.1)
y = np.arange(0.04, 0.06, 0.001)
X,Y = np.meshgrid(x, y)
Z = []
for b in y:
    for a in x:
        Z.append(simple_t_rev(a, b))
Z = np.array(Z)
Z = Z.reshape((len(y), len(x)))
Z2 = simple_delta_over_b(X, Y)
Z3 = simple_T_req(X, Y)

plt.xlabel(r'AR')
plt.ylabel(r'S ($m^2$)')
plt.title('Optimization of $t_{rev}$ ($C_{L} = 1$)')

plt.plot(5.14, 0.0499, 'k.')
plt.plot(6.26, 0.0477, 'k.')
plt.plot(7.04, 0.0467, 'k.')

C1 = plt.contour(X, Y, Z, levels=[6.0, 6.2, 6.4, 6.6, 6.8, 7.0], colors = '0.7')
plt.clabel(C1, inline=1, fontsize=10, fmt='%1.1f')
C2 = plt.contour(X, Y, Z2, inline=1, fontsize=10, levels=[0.05, 0.1, 0.15], colors=['r', 'g', 'b'])
CB2 = plt.colorbar(C2, shrink=0.8, extend='both', pad = 0.05, label=r"$\frac{\delta}{b}_{max}$")
C3 = plt.contour(X, Y, Z3, levels=[0.7], colors = 'r')

plt.annotate(r"$t_{rev}$ = 6.67 s", xy=(5.14, 0.0499), xytext=(4.5, 0.055),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.37 s", xy=(6.26, 0.0477), xytext=(5.6, 0.0545),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$t_{rev}$ = 6.22 s", xy=(7.04, 0.0467), xytext=(6.2, 0.056),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$T_{max}$ = 0.7 N", xy=(5.4, 0.0493), xytext=(5.1, 0.0445),
             arrowprops=dict(arrowstyle="->"))

plt.show(block=False)

input("Continue...")
plt.close()

# Graph 1D t_rev with d/b points

plt.axhline(y=0, color='0.7', linestyle='dashed')
plt.plot(6.67, 0, 'r.')
plt.plot(6.37, 0, 'g.')
plt.plot(6.22, 0, 'b.')

plt.xlabel(r'$t_{rev}$')
plt.title(r'$t_{rev}$ optimization results')

plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.05", xy=(6.67, 0), xytext=(6.57, 0.01),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.1", xy=(6.37, 0), xytext=(6.2, 0.01),
             arrowprops=dict(arrowstyle="->"))
plt.annotate(r"$\frac{\delta}{b}_{max}$ = 0.15", xy=(6.22, 0), xytext=(5.77, 0.01),
             arrowprops=dict(arrowstyle="->"))

plt.gca().axes.get_yaxis().set_visible(False)
plt.xlim([5.5, 7])
plt.show(block=False)

input("Continue...")
plt.close()
