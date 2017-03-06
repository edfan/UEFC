#! /usr/bin/python3

class Plane():
    """
    Variable convention:
    Parameters are stored in the object using their names (listed below).
    A method to calculate a parameter has "c_" appended to the start of the
    parameter name. These methods do NOT set the result into the Plane object;
    if wanted, set it explicitly. These methods do not take any arguments;
    they read from the Plane object.

    Ex. Calculating and setting W_wing:
    p = Plane(...various required parameters)
    W_wing = p.c_W_wing() # Calculates W_wing, but does not set it in p.
    p.W_wing = p.c_W_wing() # Calculates and sets W_wing.

    Parameter names:
    A: cross-sectional area
    CDA_0: 
    e: Span efficiency factor
    eps: 
    E_foam:
    rho_foam: foam density
    tau:
    T_max: Maximum available thrust
    W_fuse: Weight of fuselage
    W_pay: Weight of payload
    W_wing: Weight of wing
    ---------
    AR: Aspect ratio
    S: 
    C_L: Coefficient of lift
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
        Returns: True if calculation is valid, False if not.
        """

        all_vars_valid = True

        for r_v in required_vars:
            if r_v not in self.__dict__.keys():
                print("Cannot calculate", to_calc, "- missing", r_v)
                all_vars_valid = False

        return all_vars_valid

    


# Represent Plane Vanilla parameters as a dictionary, which can then be turned
# into a Plane object.
plane_vanilla_params = {
    'CDA_0': 0.004,
    'e': 0.95,
    'tau': 0.11,
    'eps': 0.03,
    'W_fuse': 2.7,
    'rho_foam': 32.0,
    'E_foam': 19.3e6
}

plane_v = Plane(**plane_vanilla_params)

