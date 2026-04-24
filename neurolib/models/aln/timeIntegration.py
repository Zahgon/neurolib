import numpy as np
import numba
from ...utils import model_utils as mu

def timeIntegration(params):
    """Sets up the parameters for time integration

    Return:
      rates_exc:  N*L array   : containing the exc. neuron rates in kHz time series of the N nodes
      rates_inh:  N*L array   : containing the inh. neuron rates in kHz time series of the N nodes
      t:          L array     : time in ms
      mufe:       N vector    : final value of mufe for each node
      mufi:       N vector    : final value of mufi for each node
      IA:         N vector    : final value of IA   for each node
      seem :      N vector    : final value of seem  for each node
      seim :      N vector    : final value of seim  for each node
      siem :      N vector    : final value of siem  for each node
      siim :      N vector    : final value of siim  for each node
      seev :      N vector    : final value of seev  for each node
      seiv :      N vector    : final value of seiv  for each node
      siev :      N vector    : final value of siev  for each node
      siiv :      N vector    : final value of siiv  for each node

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """
    pass

@numba.njit(locals={'idxX': numba.int64, 'idxY': numba.int64, 'idx1': numba.int64, 'idy1': numba.int64})
def timeIntegration_njit_elementwise(dt, duration, distr_delay, filter_sigma, Cmat, Dmat, c_gl, Ke_gl, tau_ou, sigma_ou, mue_ext_mean, mui_ext_mean, sigmae_ext, sigmai_ext, Ke, Ki, de, di, tau_se, tau_si, tau_de, tau_di, cee, cie, cii, cei, Jee_max, Jei_max, Jie_max, Jii_max, a, b, EA, tauA, C, gL, EL, DeltaT, VT, Vr, Vs, Tref, taum, mufe, mufi, IA, seem, seim, seev, seiv, siim, siem, siiv, siev, precalc_r, precalc_V, precalc_tau_mu, precalc_tau_sigma, dI, ds, sigmarange, Irange, N, Dmat_ndt, t, rates_exc, rates_inh, rd_exc, rd_inh, sqrt_dt, startind, ndt_de, ndt_di, mue_ou, mui_ou, ext_exc_rate, ext_inh_rate, ext_exc_current, ext_inh_current, noise_exc, noise_inh):
    pass

@numba.njit(locals={'idxX': numba.int64, 'idxY': numba.int64})
def interpolate_values(table, xid1, yid1, dxid, dyid):
    pass

@numba.njit(locals={'idxX': numba.int64, 'idxY': numba.int64})
def lookup_no_interp(x, dx, xi, y, dy, yi):
    """
    Return the indices for the closest values for a look-up table
    Choose the closest point in the grid

    x     ... range of x values
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0])
               (same for y)

    return:   idxX and idxY
    """
    pass

@numba.njit(locals={'xid1': numba.int64, 'yid1': numba.int64, 'dxid': numba.float64, 'dyid': numba.float64})
def fast_interp2_opt(x, dx, xi, y, dy, yi):
    """
    Returns the values needed for interpolation:
    - bilinear (2D) interpolation within ranges,
    - linear (1D) if "one edge" is crossed,
    - corner value if "two edges" are crossed

    x     ... range of the x value
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0] )
    (same for y)

    return:   xid1    ... index of the lower interpolation value
              dxid    ... distance of xi to the lower interpolation value
              (same for y)
    """
    pass

@numba.njit
def jacobian_aln(model_params, precomp_factors, V, fullstate, ue, ui, ure, uri, nw_input, nw_input_sq, re_del, ri_del, sv):
    """Jacobian of the ALN dynamical system.

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param V:                   Number of system variables.
    :type V:                    int
    :param fullstate:           Value of all V=16 dynamical variables at given time
    :type fullstate:            np.ndarray
    :param ue:                  Control input to E population
    :type ue:                   float
    :param ui:                  Control input to I population
    :type ui:                   float
    :param  nw_input:           sum of all network inputs into current node at current time
    :type  nw_input:            float
    :param  nw_input_sq:        sum of all network inputs into current node at current time with squared prefactors
    :type  nw_input_sq:         float
    :param re_del:              E rate delayed by de
    :type re_del:               float
    :param ri_del:              I rate delayed by di
    :type ri_del:               float
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    V x V Jacobian matrix.
    :rtype:                     np.ndarray
    """
    pass

def compute_hx(model_params, precomp_factors, N, V, T, dyn_vars, control, cmat, dmat_ndt, ndt_de, ndt_di, sv):
    """Jacobian of the ALN dynamical system.

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Time-dependent values of all N x V dynamical variables (all nodes)
    :type dyn_vars:             np.ndarray
    :param control:             Control input (time dependent and all input channels)
    :type control:              np.ndarray
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints
    :param ndt_de:              E rate delay in time steps
    :type ndt_de:               int
    :param ndt_di:              I rate delay in time steps
    :type ndt_di:               int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    N x T x V x V Jacobian matrix.
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def compute_nw_input(N, T, re, cmat, dmat_ndt, c_gl, Ke_gl):
    pass

@numba.njit
def jacobian_de(model_params, precomp_factors, V, fullstate, ue, ui, ure, uri, nw_input, re_del, ri_del, sv):
    """Jacobian of the ALN dynamical system wrt relations with delay de

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param V:                   Number of system variables.
    :type V:                    int
    :param fullstate:           Value of all V=16 dynamical variables at given time
    :type fullstate:            np.ndarray
    :param ue:                  Control input to E population
    :type ue:                   float
    :param ui:                  Control input to I population
    :type ui:                   float
    :param  nw_input:           sum of all network inputs into current node at current time
    :type  nw_input:            float
    :param re_del:              E rate delayed by de
    :type re_del:               float
    :param ri_del:              I rate delayed by di
    :type ri_del:               float
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    V x V Jacobian matrix of delayed variables
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def compute_hx_de(model_params, precomp_factors, N, V, T, dyn_vars, control, cmat, dmat_ndt, ndt_de, ndt_di, sv):
    """Jacobian of the ALN dynamical system wrt variables delayed by de

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Time-dependent values of all N x V dynamical variables (all nodes)
    :type dyn_vars:             np.ndarray
    :param control:             Control input (time dependent and all input channels)
    :type control:              np.ndarray
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints
    :param ndt_de:              E rate delay in time steps
    :type ndt_de:               int
    :param ndt_di:              I rate delay in time steps
    :type ndt_di:               int

    :return:                    N x T x V x V Jacobian matrix of delayed variables
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def jacobian_di(model_params, precomp_factors, V, fullstate, ue, ui, ure, uri, nw_input, re_del, ri_del, sv):
    """Jacobian of the ALN dynamical system wrt relations with delay di
    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param V:                   Number of system variables.
    :type V:                    int
    :param fullstate:           Value of all V=16 dynamical variables at given time
    :type fullstate:            np.ndarray
    :param ue:                  Control input to E population
    :type ue:                   float
    :param ui:                  Control input to I population
    :type ui:                   float
    :param  nw_input:           sum of all network inputs into current node at current time
    :type  nw_input:            float
    :param re_del:              E rate delayed by de
    :type re_del:               float
    :param ri_del:              I rate delayed by di
    :type ri_del:               float
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    V x V Jacobian matrix of delayed variables
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def compute_hx_di(model_params, precomp_factors, N, V, T, dyn_vars, control, cmat, dmat_ndt, ndt_de, ndt_di, sv):
    """Jacobian of the ALN dynamical system wrt variables delayed by di

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Time-dependent values of all N x V dynamical variables (all nodes)
    :type dyn_vars:             np.ndarray
    :param control:             Control input (time dependent and all input channels)
    :type control:              np.ndarray
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints
    :param ndt_de:              E rate delay in time steps
    :type ndt_de:               int
    :param ndt_di:              I rate delay in time steps
    :type ndt_di:               int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    N x T x V x V Jacobian matrix of delayed variables
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def compute_hx_nw(model_params, precomp_factors, N, V, T, dyn_vars, control, cmat, dmat_ndt, ndt_de, ndt_di, sv):
    """Jacobian of the ALN dynamical system wrt network connections

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Time-dependent values of all N x V dynamical variables (all nodes)
    :type dyn_vars:             np.ndarray
    :param control:             Control input (time dependent and all input channels)
    :type control:              np.ndarray
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints
    :param ndt_de:              E rate delay in time steps
    :type ndt_de:               int
    :param ndt_di:              I rate delay in time steps
    :type ndt_di:               int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    N x N x T x V x V Jacobian matrix of network connections
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def jacobian_nw(model_params, precomp_factors, V, fullstate, re_del, ri_del, nw_input, cmat_entry, ue, ure, sv):
    """Jacobian of the ALN dynamical system wrt network connections

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param V:                   Number of system variables.
    :type V:                    int
    :param fullstate:           Value of all V=16 dynamical variables at given time
    :type fullstate:            np.ndarray
    :param re_del:              E rate delayed by de
    :type re_del:               float
    :param ri_del:              I rate delayed by di
    :type ri_del:               float
    :param  nw_input:           sum of all network inputs into current node at current time
    :type  nw_input:            float
    :param cmat_entry:          Entry of the connectivity matrix at n1, n2
    :type cmat_entry:           float
    :param ue:                  Control input to E population
    :type ue:                   float
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    V x V Jacobian matrix of network variables
    :rtype:                     np.ndarray
    """
    pass

@numba.njit
def Duh(model_params, precomp_factors, N, V_in, V_vars, T, fullstate, cmat, dmat_ndt, ue, ui, ure, uri, sv):
    """Derivative of systems dynamics wrt. external inputs (control signals).

    :param model_params:    Ordered tuple of parameters in the ALNModel in order
    :type model_params:     tuple of float and np.ndarray
    :param precomp_factors:     Ordered tuple of precomputed factors required repeatedly in the computation
    :type precomp_factors:      tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V_in:                Number of input channels (control channels).
    :type V_in:                 int
    :param V_vars:              Number of dynamical variables.
    :type V_vars:               int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param fullstate:           Time-dependent values of all N x V dynamical variables (all nodes)
    :type fullstate:            np.ndarray
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints

    :return:    N x V x V x T matrix
    :rtype:     np.ndarray
    """
    pass

@numba.njit
def Dxdoth(N, V, sv):
    """Derivative of system dynamics wrt x dot

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of system variables.
    :type V:        int

    :return:        N x V x V matrix.
    :rtype:         np.ndarray
    """
    pass