from copy import deepcopy
import numpy as np
import symengine as se
from jitcdde import y as state_vector
from .....utils.stimulus import Input

class NeuralMass:
    """
    Represents a neural population with given parameters and equations, in
    particular, the derivatives of the state vector.
    """
    name = ''
    label = ''
    mass_type = None
    num_state_variables = 0
    index = None
    coupling_variables = {}
    required_couplings = []
    num_noise_variables = 0
    _noise_input = []
    state_variable_names = []
    required_params = []
    helper_variables = []
    python_callbacks = []
    noise_input_idx = None
    DESCRIPTION_FIELD = ['index', 'name', 'mass_type', 'num_state_variables', 'num_noise_variables', 'state_variable_names', 'params']

    def __init__(self, params, seed=None):
        """
        :param params: parameters of the neural mass
        :type params: dict
        :param seed: seed for random number generator
        :type seed: int|None
        """
        assert isinstance(params, dict)
        self.params = deepcopy(params)
        self.seed = seed
        self.idx_state_var = None
        self.initialised = False
        self.helper_symbols = {symbol: se.Symbol(symbol) for symbol in self.helper_variables}
        self.callback_functions = {function: se.Function(function) for function in self.python_callbacks}
        self._validate_params()

    def __str__(self):
        """
        String representation.
        """
        return f"Neural mass: {self.name} with {self.num_state_variables} state variables: {', '.join(self.state_variable_names)}"

    def __repr__(self):
        return self.__str__()

    def describe(self):
        """
        Return description dict.
        """
        pass

    def _initialize_state_vector(self):
        """
        Initialize state vector. By default it is all zeroes.
        """
        pass

    def _validate_params(self):
        """
        Validate parameters - check if self.params contains all required
        parameters.
        """
        pass

    def _validate_callbacks(self, callback_list):
        """
        Validate callbacks - mainly the length and symbolic function names.

        :param callback_list: list of callbacks
        :type callback_list: list[tuple|list]
        """
        pass

    @property
    def noise_input(self):
        pass

    @noise_input.setter
    def noise_input(self, new_noise):
        pass

    def init_mass(self, start_idx_for_noise=None):
        """
        Initialise neural mass. Usually just initialise the state vector,
        possibly can be subclassed and do other initialisation.
        """
        pass

    def _get_params_from_noise(self):
        pass

    def update_params(self, params_dict, **kwargs):
        """
        Update parameters of the mass.

        :param params_dict: new parameters for this mass
        :type params_dict: dict
        """
        pass

    def _callbacks(self):
        """
        List of python callbacks within the symbolic derivatives definition. By
        default, return empty list. If needed, redefine in subclass.

        NOTE: if you would like to use `numba` backend, the callbacks need to be
        defined as so-called jitted functions, i.e. they cannot be class methods
        but rather basic functions that are wrapped with `@numba.njit()`
        """
        pass

    def _numba_callbacks(self):
        """
        List of python callbacks (see above) for numba integrator. By default,
        returns the same callbacks as for symbolic, redefine if this need to be
        different.
        """
        pass

    def _unwrap_state_vector(self):
        """
        Unwrap state vector into individual variables. Uses global
        `state_vector` from `jitc*de`.
        """
        pass

    def _derivatives(self, coupling_variables=None):
        """
        Define derivates, i.e. right-hand side of the dynamical equation
        describing dynamics of this neural mass. Optional input is
        `coupling variables`, should return a list of derivatives of the state
        vector (of the same length obviously).

        :param coupling_variables: optional coupling variables for the mass, as
            a dictionary {"coupling variable name": symengine.Function}
        :type coupling_variables: dict|None
        :return: derivatives of the state vector
        :rtype: list
        """
        pass