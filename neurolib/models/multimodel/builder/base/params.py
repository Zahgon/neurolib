"""
Set of convenience functions for parameter handling in MultiModel.
"""
import numpy as np
import sympy as sp
from sympy.core import symbol

def count_float_params(param_dict):
    """
    Count number of float parameters in a dictionary, do not count noise
    parameters.
    """
    pass

def float_params_to_vector_symbolic(param_dict):
    """
    Transforms float / array / int parameters to symbolic ones. Assumes flat
    dictionary with dot as a separator. Does not translate noise parameters.

    :param param_dict: dictionary with parameters and their values
    :type param_dict: dict
    :return: dictionary with parameters and symbols
    :rtype: dict
    """
    pass

def float_params_to_individual_symbolic(param_dict):
    """
    Transforms float / array / int parameters to symbolic ones. Assumes flat
    dictionary with dot as a separator. Does not translate noise parameters. All
    parameters are Symbols, compatible with symengine.
    """
    pass