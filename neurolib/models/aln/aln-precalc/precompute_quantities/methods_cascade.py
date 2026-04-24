# -*- coding: utf-8 -*-
"""
 functions to calculate the quantities required by the LN cascade models,
 including functions to compute the steady-state and the first order rate response
 of an exponential/leaky integrate-and-fire neuron subject to white noise input
 (and modulations of the input moments) -- written by Josef Ladenbauer in 2016

 Comment from neurolib-dev: Installing neurolib does not install the requirements
 of this file.
"""

import numpy as np
import scipy.optimize
import numba
import multiprocessing
import time
import tables
from warnings import warn

# COMPUTING FUNCTIONS ---------------------------------------------------------

# prepares data structures and calls computing functions (possibly in parallel)
def calc_EIF_output_and_cascade_quants(
    mu_vals, sigma_vals, params, EIF_output_dict, output_names, save_rate_mod, LN_quantities_dict, quantity_names
):

    pass


# wrapper function that calls computing functions for a given sigma value and
# looping over all given mu values (depending on what needs to be computed)
def output_and_quantities_given_sigma_wrapper(arg_tuple):
    pass


# CORE FUNCTIONS that calculate steady state and 1st order spike rate response
# to modulations for an EIF/LIF neuron subject to white noise input


@numba.njit
def EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma):
    pass


@numba.njit
def EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref, mu, sigma, inhom, w_vec):
    pass


def EIF_find_lin_response_peak(
    w_vec, r1_vec, r1_f0, V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref, mu, sigma, inhom, abs_re_im
):
    pass


@numba.njit
def EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref, mu, sigma, inhom, w):
    pass


# the functions above work efficiently in practice, but the integration schemes
# might be improved (e.g., based on the Magnus expansion) to allow for larger
# membrane voltage discretization steps


def fit_exponential_freqdom(f, r1_mod_normalized, init_val):

    pass


def exp_mean_sq_dist(tau, *args):
    pass


def fit_exp_damped_osc_freqdom(
    init_vals, fpeak_real_r1_mumod, peak_real_r1_mumod, fpeak_imag_r1_mumod, peak_imag_r1_mumod, firstfit
):

    # first global brute-force optimization on a coarse grid to reduce risk of
    # finding a local optimum (narrow ranges used here)
    # Note that some of the values set here might not be optimal for certain
    # parametrizations of the EIF/LIF model
    pass


def dosc_mean_sq_dist_2f(p, *args):
    pass


@numba.njit
def dosc_mean_sq_dist_2f_tauf0grid(tau_vals, f0_vals, args):
    pass


@numba.njit
def eval_dosc_fdom(p, f):
    pass


# LOAD / SAVE FUNCTIONS --------------------------------------------------------


def load(filepath, input_dict, quantities, param_dict):
    pass


def save(filepath, output_dict, param_dict):
    pass
