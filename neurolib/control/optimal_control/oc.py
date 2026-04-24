import abc
import numba
import numpy as np
from neurolib.control.optimal_control import cost_functions
from neurolib.utils.model_utils import computeDelayMatrix, adjustArrayShape
import logging
import copy
from numba.core import types
from numba.typed import Dict

def getdefaultweights():
    pass

@numba.njit
def compute_gradient(N, V, dim_out, df_du, adjoint_state, control_matrix, d_du, control_interval):
    """Compute the gradient of the total cost wrt. the control signals (explicitly and implicitly) given the adjoint
       state, the Jacobian of the total cost wrt. explicit control contributions and the Jacobian of the dynamics
       wrt. explicit control contributions.

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of  variables of the model.
    :type V:        int
    :param dim_out: Number of 'output variables' of the model.
    :type dim_out:  int
    :param T:       Length of simulation (time dimension).
    :type T:        int
    :param df_du:   Derivative of the cost wrt. the explicit control contributions to cost functionals.
    :type df_du:    np.ndarray of shape N x V x T
    :param adjoint_state:  Solution of the adjoint equation.
    :type adjoint_state:   np.ndarray of shape N x V x T
    :param control_matrix: Binary matrix that defines nodes and variables where control inputs are active, defaults to
                           None.
    :type control_matrix:  np.ndarray of shape N x V
    :param d_du:     Jacobian of systems dynamics wrt. the external inputs (control).
    :type d_du:      np.ndarray of shape V x V
    :return:         The gradient of the total cost wrt. the control.
    :rtype:          np.ndarray of shape N x V x T
    """
    pass

@numba.njit
def solve_adjoint(hx_list, del_list, hx_nw, fx, state_dim, dt, N, T, dmat_ndt, dxdoth, state_vars, output_vars):
    """Backwards integration of the adjoint state.

    :param hx_list:     list of Jacobians of systems dynamics wrt. 'state_vars'
    :type hx_list:      list of np.ndarray
    :param del_list:    list of respective time delay integer
    :type del_list:     list of int
    :param hx_nw:       Jacobians for each time step for the network coupling.
    :type hx_nw:        np.ndarray
    :param fx: df/dx    Derivative of cost function wrt. systems dynamics.
    :type fx:           np.ndarray
    :param state_dim:   Dimensions of state (N, V, T).
    :type state_dim:    tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dmat_ndt:    N x N delay matrix (discrete number of delayed time-intervals).
    :type dmat_ndt:     np.ndarray
    :param dxdoth:      derivative of system dynamics wrt x dot
    :type dxdoth:       np.ndarray
    :param state_vars:      list of state variables of model
    :type state_vars:       list
    :param output_vars:     list of output variables of model
    :type output_vars:      list

    :return:            Adjoint state.
    :rtype:             np.ndarray of shape `state_dim`
    """
    pass

@numba.njit
def adjoint_input(hx_list, del_list, t, T_lim, state_dim1, adj, n, k):
    """Compute input to adjoint state for backwards integration

    :param hx_list:     list of Jacobians of systems dynamics wrt. 'state_vars'
    :type hx_list:      list of np.ndarray
    :param del_list:    list of respective time delay integer
    :type del_list:     list of int
    :param t:           current time index
    :type t:            int
    :param T_lim:       Maximum time index
    :type T_lim:        int
    :param state_dim1:  Number of state variables (V)
    :type state_dim1:   int
    :param adj:         adjoint state
    :type adj:          np.ndarray
    :param n:           node index
    :type n:            int
    :param k:           node index
    :type k:            int

    :return:            Adjoint state input
    :rtype:             float
    """
    pass

@numba.njit
def adjoint_nw_input(N, n, k, dmat_ndt, t, T_lim, state_dim1, adj, hxnw):
    """Compute input to adjoint state from network connections for backwards integration

    :param N:           Number of nodes in the network.
    :type N:            int
    :param n:           current node index
    :type n:            int
    :param k:           node index
    :type k:            int
    :param dmat_ndt:    N x N delay matrix (discrete number of delayed time-intervals).
    :type dmat_ndt:     np.ndarray
    :param t:           current time index
    :type t:            int
    :param T_lim:       Maximum time index
    :type T_lim:        int
    :param state_dim1:  Number of state variables (V)
    :type state_dim1:   int
    :param adj:          adjoint state
    :type adj:           np.ndarray
    :param hxnw:        Jacobians for each time step for the network coupling.
    :type hx_w:         np.ndarray

    :return:            Adjoint state input
    :rtype:             float
    """
    pass

@numba.njit
def limit_control_to_interval(N, dim_in, T, control, control_interval):
    pass

@numba.njit
def update_control_with_limit(N, dim_in, T, control, step, gradient, u_max):
    """Computes the updated control signal. The absolute values of the new control are bounded by +/- 'u_max'. If
       'u_max' is 'None', no limit is applied.

    :param control:         N x V x T array. Control signals.
    :type control:          np.ndarray
    :param step:            Step size along the gradients.
    :type step:             float
    :param gradient:        N x V x T array of the gradients.
    :type gradient:         np.ndarray
    :param u_max:           Maximum absolute value allowed for the strength of the control signal.
    :type u_max:            float or None

    :return:                N x V x T array containing the new control signal, updated according to 'step' and
                            'gradient' with the maximum absolute values being limited by 'u_max'.
    :rtype:                 np.ndarray
    """
    pass

def convert_interval(interval, array_length):
    """Turn indices into positive values only. It is assumed in any case, that the first index defines the start and
       the second the stop index, both inclusive.

    :param interval:    Tuple containing start and stop index. May contain negative indices or 'None'.
    :type interval:     tuple
    :param array_length:    Length of the array in the dimension, along which the 'interval' is defining the slice.
    :type array_length:     int
    :return:            Tuple containing two positive 'int' indicating the start- and stop-index (both inclusive) of the
                        interval.
    :rtype:             tuple
    """
    pass

class OC:

    def __init__(self, model, target, weights=None, maximum_control_strength=None, print_array=[], cost_interval=(None, None), control_interval=(None, None), cost_matrix=None, control_matrix=None, M=1, M_validation=0, validate_per_step=False):
        """
        Base class for optimal control. Model specific methods should be implemented in derived class for each model.

        :param model:       An instance of neurolib's Model-class. Parameters like '.duration' and methods like '.run()'
                            are used within the optimal control.
        :type model:        neurolib.models.model
        :param target:      Target time series of controllable variables.
        :type target:       np.ndarray
        :param weights:     Dictionary of weight parameters, defaults to 'None'.
        :type weights:      dictionary, optional
        :param maximum_control_strength:    Maximum absolute value a control signal can take. No limitation of the
                                            absolute control strength if 'None'. Defaults to None.
        :type:                              float or None, optional
        :param print_array:                 Array of optimization-iteration-indices (starting at 1) in which cost is printed out.
                                            Defaults to empty list `[]`.
        :type print_array:                  list, optional
        :param cost_interval:               (t_start, t_end). Indices of start and end point (both inclusive) of the
                                            time interval in which the accuracy cost is evaluated. Default is full time
                                            series. Defaults to (None, None).
        :type cost_interval:                tuple, optional
        :param control_interval:            (t_start, t_end). Indices of start and end point (both inclusive) of the
                                            time interval in which control can be applied. Default is full time
                                            series. Defaults to (None, None).
        :type control_interval:              tuple, optional
        :param cost_matrix:                 N x V binary matrix that defines nodes and channels of accuracy measurement, defaults
                                            to None.
        :type cost_matrix:                  np.ndarray
        :param control_matrix:      N x V Binary matrix that defines nodes and variables where control inputs are active,
                                    defaults to None.
        :type control_matrix:       np.ndarray
        :param M:                   Number of noise realizations. M=1 implies deterministic case. Defaults to 1.
        :type M:                    int, optional
        :param M_validation:        Number of noise realizations for validation (only used in stochastic case, M>1).
                                    Defaults to 0.
        :type M_validation:         int, optional
        :param validate_per_step:   True for validation in each iteration of the optimization, False for
                                    validation only after final optimization iteration (only used in stochastic case,
                                    M>1). Defaults to False.
        :type validate_per_step:    bool, optional

        """
        self.model = copy.deepcopy(model)
        self.target = target
        self.maximum_control_strength = maximum_control_strength
        if type(weights) != type(dict()):
            if weights is not None:
                print('Weights parameter must be dictionary, use default weights instead.')
            self.weights = getdefaultweights()
        else:
            defaultweights = getdefaultweights()
            for k in defaultweights.keys():
                if k in weights.keys():
                    defaultweights[k] = weights[k]
                else:
                    print('Weight ', k, ' not in provided weight dictionary. Use default value.')
            self.weights = defaultweights
        self.N = self.model.params.N
        self.dt = self.model.params['dt']
        self.duration = self.model.params['duration']
        self.T = np.around(self.duration / self.dt, 0).astype(int) + 1
        self.dim_vars = len(self.model.state_vars)
        self.dim_in = len(self.model.input_vars)
        self.dim_out = len(self.model.output_vars)
        self.state_vars_dict = self.get_state_vars_dict()
        self.adjust_init()
        self.simulate_forward()
        if self.N == 1:
            self.Dmat_ndt = np.zeros((self.N, self.N)).astype(int)
        else:
            Dmat = computeDelayMatrix(self.model.params.lengthMat, self.model.params.signalV)
            if self.model.name != 'aln':
                Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
            else:
                Dmat[np.eye(len(Dmat)) == 1] = np.ones(len(Dmat)) * self.model.params.de
            self.Dmat_ndt = np.around(Dmat / self.dt).astype(int)
        if self.N == 1:
            if isinstance(self.model.Cmat, type(None)):
                self.model.Cmat = np.zeros((self.N, self.N))
            if isinstance(self.model.Dmat, type(None)):
                self.model.Dmat = np.zeros((self.N, self.N))
        self.cost_matrix = cost_matrix
        if isinstance(self.cost_matrix, type(None)):
            self.cost_matrix = np.ones((self.N, self.dim_out))
        self.control_matrix = control_matrix
        if isinstance(self.control_matrix, type(None)):
            self.control_matrix = np.ones((self.N, self.dim_in))
        self.M = max(1, M)
        self.M_validation = M_validation
        self.step = 10.0
        self.count_noisy_step = 10
        self.count_step = 30
        self.factor_down = 0.5
        self.factor_up = 2.0
        self.cost_validation = 0.0
        self.validate_per_step = validate_per_step
        if self.model.params.sigma_ou != 0.0:
            if self.M <= 1:
                logging.warning('For noisy system, please chose parameter M larger than 1 (recommendation > 10).' + '\n' + 'If you want to study a deterministic system, please set model parameter "sigma_ou" to zero')
            if self.M > self.M_validation:
                print('Parameter "M_validation" should be chosen larger than parameter "M".')
        elif self.M > 1 or self.M_validation != 0 or validate_per_step:
            print('For deterministic systems, parameters "M", "M_validation" and "validate_per_step" are not relevant.' + '\n' + 'If you want to study a noisy system, please set model parameter "sigma_ou" larger than zero')
        self.state_dim = (self.N, self.dim_vars, self.T)
        self.adjoint_state = np.zeros(self.state_dim)
        self.gradient = np.zeros(self.state_dim)
        self.cost_history = []
        self.step_sizes_history = []
        self.step_sizes_loops_history = []
        self.control_history = []
        self.print_array = print_array
        self.zero_step_encountered = False
        self.cost_interval = convert_interval(cost_interval, self.T)
        self.control_interval = convert_interval(control_interval, self.T)
        self.ndt_de, self.ndt_di = (0.0, 0.0)
        self.adjust_input()
        control = np.zeros((self.N, self.dim_in, self.T))
        for v, iv in enumerate(self.model.input_vars):
            control[:, v, :] = self.model.params[iv]
        self.control = control.copy()
        self.check_params()
        self.control = update_control_with_limit(self.N, self.dim_in, self.T, control, 0.0, np.zeros(control.shape), self.maximum_control_strength)
        self.model_params = self.get_model_params()

    def check_params(self):
        """Checks a subset of parameters and throws an error if a wrong dimension is found."""
        pass

    def get_state_vars_dict(self):
        """Creates a numba dictionary which maps the state variable names with their indices."""
        pass

    def adjust_init(self):
        """Adjust the shape of the array provided as init to the model. Use adjustArrayShape function from model_utils."""
        pass

    def adjust_input(self):
        """Adjust the shape of the array provided as input to the model. Use adjustArrayShape function from model_utils."""
        pass

    def get_xs(self):
        """Extract the complete state of the dynamical system."""
        pass

    def get_xs_delay(self):
        """Extract the complete state of the delayed dynamical system."""
        pass

    def update_input(self):
        """Update the parameters in 'self.model' according to the current control such that 'self.simulate_forward'
        operates with the appropriate control signal.
        """
        pass

    def simulate_forward(self):
        """Updates 'state_vars' of 'self.model' in accordance to the current 'self.control'. Results for the controllable state
        variables can be accessed with self.get_xs()
        """
        pass

    @abc.abstractmethod
    def get_model_params(self):
        """Model params as an ordered tuple"""
        pass

    @abc.abstractmethod
    def Dxdot(self):
        """V x V Jacobian of systems dynamics wrt. change of all 'state_vars'."""
        pass

    @abc.abstractmethod
    def Duh(self):
        """Jacobian of systems dynamics wrt. external inputs (control signals) to all 'state_vars'."""
        pass

    def compute_total_cost(self):
        """Compute the total cost as weighted sum precision of all contributing cost terms.
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def compute_gradient(self):
        """Compute the gradient of the total cost wrt. the control:
        1. solve the adjoint equation backwards in time
        2. compute derivatives of cost wrt. control
        3. compute Jacobians of the dynamics wrt. control
        4. compute gradient of the cost wrt. control(i.e., negative descent direction)

        :return:        The gradient of the total cost wrt. the control.
        :rtype:         np.ndarray of shape N x V x T
        """
        pass

    @abc.abstractmethod
    def compute_hx_list(self):
        """Jacobians of model dynamics wrt. its 'state_vars' at each time step."""
        pass

    @abc.abstractmethod
    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling."""
        pass

    @abc.abstractmethod
    def compute_dxdoth(self):
        pass

    def solve_adjoint(self):
        """Backwards integration of the adjoint state."""
        pass

    def decrease_step(self, cost, cost0, step, control0, factor_down, cost_gradient):
        """Find a step size which leads to improved cost given the gradient. The step size is iteratively decreased.
        The control-inputs are updated in place according to the found step size via the
        "self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_down:  Factor the step size is scaled with in each iteration until cost is improved.
        :type factor_down:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        pass

    def increase_step(self, cost, cost0, step, control0, factor_up, cost_gradient):
        """Find the largest step size which leads to the biggest improvement of cost given the gradient. The step size is
        iteratively increased. The control-inputs are updated in place according to the found step size via the
        "self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_up:  Factor the step size is scaled with in each iteration while the cost keeps improving.
        :type factor_up:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        pass

    def step_size(self, cost_gradient):
        """Adaptively choose a step size for control update.

        :param cost_gradient:   N x V x T gradient of the total cost wrt. control.
        :type cost_gradient:    np.ndarray

        :return:    Step size that got multiplied with the 'cost_gradient'.
        :rtype:     float
        """
        pass

    def optimize(self, n_max_iterations):
        """Optimization method
            Choose deterministic (M=1 noise realizations) or stochastic (M>1 noise realizations) approach.
            The control-inputs are updated in place throughout the optimization.

        :param n_max_iterations: Maximum number of iterations of gradient descent.
        :type n_max_iterations:  int
        """
        pass

    def optimize_deterministic(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 0 (deterministic, M=1).

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """
        pass

    def optimize_noisy(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 3.

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """
        pass

    def compute_cost_noisy(self, M):
        """Computes the average cost from 'M_validation' noise realizations.

        :param M:                   Number of noise realizations. M=1 implies deterministic case. Defaults to 1.
        :type M:                    int, optional

        :rtype: float
        """
        pass