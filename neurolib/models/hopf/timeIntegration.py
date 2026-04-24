import numpy as np
import numba
from ...utils import model_utils as mu

def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """
    pass

@numba.njit
def timeIntegration_njit_elementwise(startind, t, dt, sqrt_dt, duration, N, Cmat, Dmat, K_gl, signalV, coupling, Dmat_ndt, xs, ys, xs_input_d, ys_input_d, x_ext, y_ext, a, w, noise_xs, noise_ys, x_ou, y_ou, x_ou_mean, y_ou_mean, tau_ou, sigma_ou):
    pass

@numba.njit
def jacobian_hopf(model_params, V, x, y, sv):
    """Jacobian of a single node of the Hopf models dynamical system wrt. its 'state_vars' ('x', 'y', 'x_ou',
       'y_ou').

    :param model_params:    Ordered tuple of parameters in the Hopf Model in order
    :type model_params:     tuple of float
    :param V:   Number of state variables.
    :type V:    int
    :param x:   Activity of x-population at this time instance.
    :type x:    float
    :param y:   Activity of y-population at this time instance.
    :type y:    float
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:    4 x 4 Jacobian matrix.
    :rtype:     np.ndarray
    """
    pass

@numba.njit
def compute_hx(model_params, K_gl, cmat, coupling, N, V, T, dyn_vars, sv):
    """Jacobians of the Hopf model wrt. its 'state_vars' at each time step.

    :param model_params:    Ordered tuple of parameters in the Hopf Model in order
    :type model_params:     tuple of float
    :param K_gl:            Model parameter of global coupling strength.
    :type K_gl:             float
    :param cmat:            Model parameter, connectivity matrix.
    :type cmat:             ndarray
    :param coupling:        Model parameter, which specifies the coupling type. E.g. "additive" or "diffusive".
    :type coupling:         str
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V:               Number of state variables.
    :type V:                int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param dyn_vars:        Time series of the activities ('x'- and 'y'-population) in all nodes. 'x' in N x 0 x T and 'y' in
                            N x 1 x T dimensions.
    :type dyn_vars:         np.ndarray of shape N x 2 x T
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                Array that contains Jacobians for all nodes in all time steps.
    :rtype:                 np.ndarray of shape N x T x 4 x 4
    """
    pass

@numba.njit
def compute_hx_nw(K_gl, cmat, N, V, T, sv):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:            Jacobians for network connectivity in all time steps.
    :rtype:             np.ndarray of shape N x N x T x 4 x 4
    """
    pass

@numba.njit
def Duh(N, V_in, V_vars, T, sv):
    """Jacobian of systems dynamics wrt. external inputs (control signals).

    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :rtype:     np.ndarray of shape N x V x V x T
    """
    pass

@numba.njit
def Dxdoth(N, V):
    """Derivative of system dynamics wrt x dot

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of system variables.
    :type V:        int

    :return:        N x V x V matrix.
    :rtype:         np.ndarray
    """
    pass