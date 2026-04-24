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
def timeIntegration_njit_elementwise(startind, t, dt, sqrt_dt, N, Cmat, K_gl, Dmat_ndt, excs, inhs, exc_input_d, inh_input_d, exc_ext_baseline, inh_ext_baseline, exc_ext, inh_ext, tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh, c_excexc, c_excinh, c_inhexc, c_inhinh, noise_exc, noise_inh, exc_ou, inh_ou, exc_ou_mean, inh_ou_mean, tau_ou, sigma_ou):
    pass

@numba.njit
def logistic(x, a, mu):
    """Logistic function evaluated at point 'x'.

    :type x:    float
    :param a:   Slope parameter.
    :type a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    pass

@numba.njit
def logistic_der(x, a, mu):
    """Derivative of logistic function, evaluated at point 'x'.

    :type x:    float
    :param a:   Slope parameter.
    :typa a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    pass

@numba.njit
def jacobian_wc(model_params, nw_e, e, i, ue, ui, V, sv):
    """Jacobian of the WC dynamical system.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param  nw_e:   N x T input of network into each node's 'exc'
    :type  nw_e:    np.ndarray
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T combined input of 'background' and 'control' into 'exc'.
    :type ue:       np.ndarray
    :param ui:      N x T combined input of 'background' and 'control' into 'inh'.
    :type ui:       np.ndarray
    :param V:       Number of system variables.
    :type V:        int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:        4 x 4 Jacobian matrix.
    :rtype:         np.ndarray
    """
    pass

@numba.njit
def compute_hx(wc_model_params, K_gl, cmat, dmat_ndt, N, V, T, dyn_vars, dyn_vars_delay, control, sv):
    """Jacobians of WCModel wrt. the 'e'- and 'i'-variable for each time step.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dyn_vars:    N x V x T array containing all values of 'exc' and 'inh'.
    :type dyn_vars:     np.ndarray
    :param dyn_vars_delay:
    :type dyn_vars_delay:     np.ndarray
    :param control:     N x 2 x T control inputs to 'exc' and 'inh'.
    :type control:      np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:            N x T x 4 x 4 Jacobians.
    :rtype:             np.ndarray
    """
    pass

@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, exc_values):
    """Compute input by other nodes of network into each node's 'exc' population at every timestep.

    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param exc_values:  N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:   np.ndarray
    :return:            N x T network inputs.
    :rytpe:             np.ndarray
    """
    pass

@numba.njit
def compute_hx_nw(model_params, K_gl, cmat, dmat_ndt, N, V, T, e, i, e_delay, ue, sv):
    """Jacobians for network connectivity in all time steps.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:       np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:         Jacobians for network connectivity in all time steps.
    :rtype:          np.ndarray of shape N x N x T x 4 x 4
    """
    pass

@numba.njit
def Duh(model_params, N, V_in, V_vars, T, ue, ui, e, i, K_gl, cmat, dmat_ndt, exc_values, sv):
    """Jacobian of systems dynamics wrt. external inputs (control signals).

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param  nw_e:           N x T input of network into each node's 'exc'
    :type  nw_e:            np.ndarray
    :param ue:              N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:               np.ndarray
    :param ui:              N x T array of the total input received by 'inh' population in every node at any time.
    :type ui:               np.ndarray
    :param e:               Value of the E-variable for each node and timepoint
    :type e:                np.ndarray
    :param i:               Value of the I-variable for each node and timepoint
    :type i:                np.ndarray
    :param K_gl:            global coupling strength
    :type K_gl              float
    :param cmat:            coupling matrix
    :type cmat:             np.ndarray
    :param dmat_ndt:        delay index matrix
    :type dmat_ndt:         np.ndarray
    :param exc_values:      N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:       np.ndarray
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