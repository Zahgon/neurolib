import numpy as np
import numba

@numba.njit
def accuracy_cost(x, target_timeseries, weights, cost_matrix, dt, interval=(0, None)):
    """Total cost related to the accuracy, weighted sum of contributions.

    :param x:               State of dynamical system.
    :type x:                np.ndarray
    :param target_timeseries:    Target state.
    :type target_timeseries:     np.darray
    :param weights:         Dictionary of weights.
    :type weights:          dictionary
    :param cost_matrix:     Matrix of channels to take into account
    :type cost_matrix:      ndarray
    :param dt:              Time step.
    :type dt:               float
    :param interval:        (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                            dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:         tuple, optional

    :return:                Accuracy cost.
    :rtype:                 float
    """
    pass

@numba.njit
def derivative_accuracy_cost(x, target_timeseries, weights, cost_matrix, interval=(0, None)):
    """Derivative of the 'accuracy_cost' wrt. the state 'x'.

    :param x:               State of dynamical system.
    :type x:                np.ndarray
    :param target_timeseries:    Target state.
    :type target_timeseries:     np.darray
    :param weights:         Dictionary of weights.
    :type weights:          dictionary
    :param cost_matrix:     Matrix of channels to take into account
    :type cost_matrix:      ndarray
    :param interval:        (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                            dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:         tuple, optional

    :return:                Accuracy cost derivative.
    :rtype:                 ndarray
    """
    pass

@numba.njit
def precision_cost(x_sim, x_target, cost_matrix, interval=(0, None)):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.
       Penalizes deviation from the target.

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param cost_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                             None.
    :type cost_matrix:  np.ndarray
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple

    :return:            Precision cost for time interval.
    :rtype:             float
    """
    pass

@numba.njit
def derivative_precision_cost(x_sim, x_target, cost_matrix, interval):
    """Derivative of 'precision_cost' wrt. 'x_sim'.

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param cost_matrix: N x V binary matrix that defines nodes and channels of precision measurement, defaults to
                        None
    :type cost_matrix:  np.ndarray
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple

    :return:            Control-dimensions x T array of precision cost gradients.
    :rtype:             np.ndarray
    """
    pass

@numba.njit
def control_strength_cost(u, weights, dt):
    """Total cost related to the control strength, weighted sum of contributions.

    :param u:           Control-dimensions x T array. Control signals.
    :type u:            np.ndarray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float

    :return:            control strength cost of the control.
    :rtype:             float
    """
    pass

@numba.njit
def derivative_control_strength_cost(u, weights, dt):
    """Derivative of the 'control_strength_cost' wrt. the control 'u'.

    :param u:           Control-dimensions x T array. Control signals.
    :type u:            np.ndarray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float

    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """
    pass

@numba.njit
def L2_cost(u):
    """'Energy' or 'L2' cost. Penalizes for control strength.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :return:    L2 cost of the control.
    :rtype:     float
    """
    pass

@numba.njit
def derivative_L2_cost(u):
    """Derivative of the 'L2_cost' wrt. the control 'u'.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """
    pass

@numba.njit
def L1D_cost_integral(u, dt):
    """'Directional sparsity' or 'L1D' cost integrated over time. Penalizes for control strength.
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return:    L1D cost of the control.
    :rtype:     float
    """
    pass

@numba.njit
def derivative_L1D_cost(u, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return :   Control-dimensions x T array of L1D-cost gradients.
    :rtype:     np.ndarray
    """
    pass