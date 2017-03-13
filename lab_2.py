#! /usr/bin/env python3

import math
import numpy as np
from scipy.optimize import minimize_scalar, brute

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
    T_max: maximum available thrust
    V: flight speed
    W: total weight
    W_max: maximum total weight supportable
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
        
    def c_c(self):
        required_vars = ['c_t', 'c_r']
        self.check_vars('c', required_vars)
        
        self.c = (self.c_t + self.c_r) / 2
        
    def c_C_D(self):
        required_vars = ['CDA_0', 'S', 'C_DP', 'C_Di']
        self.check_vars('C_D', required_vars)
        
        self.C_D = self.CDA_0 / self.S + self.C_DP + self.C_Di
            
    def c_C_DP(self):
        required_vars = ['c_d0', 'c_d1', 'c_d2', 'c_d8', 'c_l', 'c_l0']
        self.check_vars('C_DP', required_vars)
        
        lift_diff = self.c_l - self.c_l0
        self.C_DP = (self.c_d0 + self.c_d1 * lift_diff + self.c_d2 * lift_diff**2 + self.c_d8 * lift_diff**8) / self.c_l
        
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

    def c_N(self):
        required_vars = ['W_fuse', 'W_wing', 'W_pay', 'rho_air', 'g', 'R', 'S', 'C_L']
        self.check_vars('N', required_vars)

        intermediate = (1 - ((self.W_fuse + self.W_wing + self.W_pay) / \
            (0.5 * self.rho_air * self.g * self.R * self.S * self.C_L))**2)
        
        if intermediate < 0:
            return 0 # Imaginary result produced otherwise.

        self.N = intermediate**-0.5

    def c_S(self):
        required_vars = ['b', 'c']
        self.check_vars('S', required_vars)

        self.S = self.b * self.c

    def c_t_rev(self):
        required_vars = ['R', 'V']
        self.check_vars('t_rev', required_vars)

        self.t_rev = 2 * math.pi * self.R / self.V

    def c_V(self):
        required_vars = ['g', 'R', 'N']
        self.check_vars('V', required_vars)

        intermediate = (self.N)**2 - 1

        if intermediate < 0:
            return 0

        self.V = (self.g * self.R * (intermediate)**0.5)**0.5 

    def c_W(self):
        required_vars = ['W_wing', 'W_fuse', 'W_pay']
        self.check_vars('W', required_vars)

        self.W = self.W_wing + self.W_fuse + self.W_pay

    def c_W_max(self):
        required_vars = ['T_max', 'C_D', 'C_L']
        self.check_vars('W_max', required_vars)

        self.W_max = self.T_max * self.C_L / self.C_D

    def c_W_pay_max(self):
        required_vars = ['W_max', 'W_wing', 'W_fuse']
        self.check_vars('W_pay_max', required_vars)

        self.W_pay_max = self.W_max - self.W_wing - self.W_fuse

    def c_W_wing(self):
        required_vars = ['rho_foam', 'g', 'tau', 'S', 'AR']
        self.check_vars('W_wing', required_vars)

        self.W_wing = 0.6 * self.rho_foam * self.g * self.tau * (self.S)**1.5 * (self.AR)**-0.5
        

# Represent Plane Vanilla parameters as a dictionary, which can then be turned
# into a Plane object.
plane_vanilla_params = {
    'b': 1.56,
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
    'g': 9.8,
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

# Find W_pay_max for delta_over_b <= 0.1.

def constrained_W_pay(W_pay, delta_over_b_max):
    p.W_pay = W_pay
    p.c_delta_over_b()
    if p.delta_over_b > delta_over_b_max:
        return 0
    return -1.0 * W_pay # Find maximum through minimum.

print('W_pay_max (constrained for d/b <= 0.1):',
      minimize_scalar(constrained_W_pay, args=(0.1), bounds=(0, 5), method='bounded').x)

# Find t_rev_min for delta_over_b = 0.1, W_pay = 0.

p.W_pay = 0
p.c_N()
p.c_V()
p.c_t_rev()

print('t_rev_min:', p.t_rev)

# Now, optimize W_pay_max

def optimized_W_pay(args, delta_over_b_max):
    p.N = 1 # back to maximum load factor
    p.S = args[0]
    p.AR = args[1]
    p.C_L = args[2]
    p.c_C_DP()
    p.c_C_Di()
    p.c_C_D()
    p.c_W_wing()
    p.c_W_max()

    p.W_pay = p.W_max - p.W_wing - p.W_fuse
    p.c_delta_over_b()
    if p.delta_over_b > delta_over_b_max:
        return 0
    return -1.0 * p.W_pay # Find maximum through minimum.

ranges = (slice(0.1, 2, 0.01), slice(1, 50, .1), slice(0.025, 1, 0.025))

print('W_pay_max (constrained for d/b <= 0.1):',
      brute(optimized_W_pay, ranges, args=(0.1,), finish=None))

print('W_pay_max (constrained for d/b <= 0.05):',
      brute(optimized_W_pay, ranges, args=(0.05,), finish=None))

print('W_pay_max (constrained for d/b <= 0.15):',
      brute(optimized_W_pay, ranges, args=(0.15,), finish=None))
    
# Optimize t_rev
def optimized_t_rev(args, delta_over_b_max):
    p.W_pay = 0
    p.S = args[0]
    p.AR = args[1]
    p.C_L = args[2]
    p.c_W_wing()
    p.c_N()
    if p.N == 0:
        return 10000
    p.c_V()
    if p.V == 0:
        return 10000
    p.c_t_rev()
    p.c_delta_over_b()
    if p.delta_over_b > delta_over_b_max:
        return 10000 # Arbitrarily large value, won't be the min.
    return p.t_rev

print('t_rev_min (constrained for d/b <= 0.1):',
      brute(optimized_t_rev, ranges, args=(0.1,), finish=None))

print('t_rev_min (constrained for d/b <= 0.05):',
      brute(optimized_t_rev, ranges, args=(0.05,), finish=None))

print('t_rev_min (constrained for d/b <= 0.15):',
      brute(optimized_t_rev, ranges, args=(0.15,), finish=None)) 
