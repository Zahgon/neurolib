"""
Backend integrator and backends definitions. Currently supported are following
backends:
 - `jitcdde`: (just-in-time compilation into C) which uses `jitcdde`
    which translates symbolic derivatives into C code and then calls C functions
    from Python interface:
    - very reliable
    - uses adaptive `dt` hence very useful for stiff problems and when you are
        not sure how stiff your model is
    - reasonable speed

 - `numba`: compilation of python code through `numba.njit()` - symbolic
    derivatives are converted to strings and prepared and compiled to
    numba-compatible python code; Equations are integrated using the Euler
    integration scheme.
    - Euler scheme could be less reliable
    - fixed integration time step `dt`
    - very fast
"""
import logging
import os
import re
import time
from copy import deepcopy
from functools import wraps
from sys import platform
from types import FunctionType
import numba
import numpy as np
import symengine as se
import sympy as sp
import xarray as xr
from chspy import CubicHermiteSpline
from jitcdde import jitcdde_input
from numpy import *
from tqdm import tqdm
from .....utils.collections import flatten_nested_dict
DEFAULT_BACKEND = 'jitcdde'

def timer(method):
    """
    Decorator for timing functions. Writes the time to logger.
    """
    pass

class BaseBackend:
    """
    Base class for backends.
    """
    _derivatives = None
    _sync = None
    _callbacks = None
    initial_state = None
    num_state_variables = None
    max_delay = None
    state_variable_names = None
    label = None
    backend_name = ''

    def run(self):
        pass

    def clean(self):
        pass

class NumbaBackend(BaseBackend):
    """
    Numba integration backend using Euler scheme with delays. The symbolic code for
    derivatives is rendered into a prepared string template using numba's njit.
    """
    backend_name = 'numba'
    DEFAULT_DT = 0.1
    CURRENT_Y_REGEX = 'current_y\\([0-9]*\\)'
    CURRENT_Y_NUMBA = 'y[{idx}, max_delay + i - 1]'
    PAST_Y_REGEX = 'past_y\\((.*?)\\)'
    PAST_Y_NUMBA = 'y[{idx}, max_delay + i - 1 - {dt_ndt}]'
    _convert_to_dt = []
    SYSTEM_INPUT_REGEX = 'past_y\\(-external_input \\+ t, ([0-9]* \\+ )?input_base_n, anchors\\(-external_input \\+ t\\)\\)'
    SYSTEM_INPUT_NUMBA = 'input_y[{idx}, i]'
    compiled_function = None
    NUMBA_EULER_TEMPLATE = '\ndef integrate(dt, system_size, max_delay, t_max, y0, input_y, {params}):\n    y = np.empty((system_size, t_max + max_delay + 1))\n    y[:] = np.nan\n    y[:, :max_delay + 1] = y0\n    for i in range(1, t_max + 1):\n        dy = np.array({dy_eqs})\n        y[:, max_delay + i] = y[:, max_delay + i - 1] + dt*dy\n\n    return y[:, max_delay + 1:]\n'

    def _replace_current_ys(self, expression):
        """
        Replace `current_y` symbolic representation of current state of the
        state vector with numpy array. Assume output as `y[time, space]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        pass

    def _replace_past_ys(self, expression, dt):
        """
        Replace `past_y` symbolic representation of past state
        vector with numpy array. Assume output as `y[time, space]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :param dt: dt for the integration
        :type dt: float
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        pass

    def _replace_inputs(self, expression):
        """
        Replace `system_input` (usually noise and/or external stimulus) symbolic
        representation with numpy array of inputs. Assume input as
        `input[time,index]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        pass

    @staticmethod
    def _substitute_helpers(derivatives, helpers):
        """
        Substitute helpers (usually used for coupling) to derivatives.

        :param derivatives: list of symbolic expressions for derivatives
        :type derivatives: list
        :param helpers: list of tuples as (helper name, symbolic expression) for
            helpers
        :type helpers: list[tuple]
        """
        pass

    @staticmethod
    def _get_numba_function_params(symbol_params):
        """
        Get all symbolic parameters as a list for numba function string template.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :return: list of all parameters as symbols
        :rtype: list[sp.Symbol]
        """
        pass

    @staticmethod
    def _create_symbol_to_float_dict(symbol_params, float_params):
        """
        Create parameters dictionary as {param_symbol: float value}.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :param float_params: float parameters as a flat dict
        :type float_params: dict
        :return: symbol: float parameters as an input to compiled numba function
        :rtype: dict
        """
        pass

    def compile_to_numba(self, symbol_params, dt, system_size):
        """
        Compile system into numba jitted nopython function.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :param dt: dt in ms
        :type dt: float
        :param system_size: number of equations in the system
        :type system_size: int
        :return: numba compiled function
        :rtype: callable
        """
        pass

    def run(self, duration, dt, noise_input, symbol_params, float_params, **kwargs):
        """
        Run integration.

        :kwargs: actually none - for compatiblity
        """
        pass

class JitcddeBackend(BaseBackend):
    """
    Backend using jitcdde integrator. Uses just-in-time compilation for delay
    differential equations with integration method proposed by Shampine and
    Thompson.

    Reference (package):
        Ansmann, G. (2018). Efficiently and easily integrating differential
        equations with JiTCODE, JiTCDDE, and JiTCSDE. Chaos: An
        Interdisciplinary Journal of Nonlinear Science, 28(4), 043116.

    Reference (method):
        Shampine, L. F., & Thompson, S. (2001). Solving DDEs in matlab. Applied
        Numerical Mathematics, 37(4), 441-458.
    """
    backend_name = 'jitcdde'
    extra_compile_args = []
    dde_system = None

    def _init_and_compile_C(self, derivatives, helpers=None, inputs=None, max_delay=0.0, callbacks=None, chunksize=1):
        """
        Initialise DDE system and compile to C.
        """
        pass

    def _set_constant_past(self, past_state, squeeze=False):
        """
        Sets past of the delayed system with a constant vector. This usually
        means that `past_state` is 1D vector of length `num_state_variables`.
        `jitcdde` automatically determines how long into the past it need to
        extrapolate based on `max_delay`.

        :param past_state: vector of past states of length `num_state_variables`
        :type past_state: np.ndarray
        """
        pass

    def _set_past_from_vector(self, past_state, dt):
        """
        Sets past of the delayed system with temporal dependance. This means
        that `past_state` is 2D array as (`num_state_variables` x `time`) and
        each time vector is added as so-called `Anchor`. `jitcdde` then
        automatically interpolate in between `dt`s when needed.

        :param past_state: vector of past states as (`num_state_variables` x
            `time`)
        :type past_state: np.ndarray
        :param dt: dt of the system, to infer how much in to the past the vector
            is (in `jitcdde` this is actually sampling dt)
        :type dt: float
        """
        pass

    def _integrate_blindly(self, max_delay):
        """
        Deals with initial discontinuities using `integrate_blindly` method,
        where the adaptive integrator just integrates until 1.5 * `max_delay`
        which smooths the initial and past discontinuities. Currently not used,
        but something is telling me this would be necessary for autochunk
        feature to work with `jitcdde` with adaptive time step.

        :param max_delay: maximum delay in the system, in ms
        :type max_delay: float
        """
        pass

    def _check(self):
        """
        Check the delay system.
        """
        pass

    def run(self, duration, dt, noise_input, **kwargs):
        """
        Run integration.
        """
        pass

    def clean(self):
        """
        Clean - i.e. remove temp directory from C compilation.
        """
        pass

class BackendIntegrator:
    """
    Backend integrator mixin - implements integration using various backends,
    stores results in xarray and is able to save xr.Datasets to pickle or
    netCDF.
    """
    backend_instance = None
    symbol_params = None
    float_params = None
    NEEDED_ATTRIBUTES = ['_derivatives', '_sync', '_callbacks', '_numba_callbacks', 'initial_state', 'num_state_variables', 'max_delay', 'state_variable_names', 'label', 'initialised']

    def _init_jitcdde_backend(self):
        pass

    def _init_numba_backend(self, dt):
        pass

    @timer
    def run(self, duration, dt, noise_input, backend=DEFAULT_BACKEND, return_xarray=True, **kwargs):
        """
        Run the integration.

        :param duration: duration of the run, in ms
        :type duration: float
        :param dt: sampling dt for `jitcdde` backend - which actually uses
            adaptive dt, or integration dt for numba backend; in ms
        :type dt: float
        :param noise_input: noise input to the network or node
        :type noise_input: `chspy.CubicHermiteSpline`|np.ndarray
        :param backend: which backend to use
        :type backend: str
        :param return_xarray: whether to return xarray's Dataset, or simply time
            and result as a matrix
        :type return_xarray: bool
        :*kwargs: optional keyword arguments, will be passed to backend instance
        """
        pass

    def _init_xarray(self, times, results):
        """
        Initialise results array.

        :param times: time for the result, in ms
        :type times: np.ndarray
        :param results: results as times x state variable
        :type results: np.ndarray
        """
        pass

    def clean(self):
        """
        Clean after myself, if needed.
        """
        pass