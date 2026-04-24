import matplotlib.pyplot as plt
import numpy as np

colors = ["red", "blue", "green", "orange"]


def plot_oc_singlenode(
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
):
    """Plot target and controlled dynamics for a single node.
    :param duration:    Duration of simulation (in ms).
    :type duration:     float
    :param dt:          Time discretization (in ms).
    :type dt:           float
    :param state:       The state of the system controlled with the found oc-input.
    :type state:        np.ndarray
    :param target:      The target state.
    :type target:       np.ndarray
    :param control:     The control signal found by the oc-algorithm.
    :type control:      np.ndarray
    :param orig_input:  The inputs that were used to generate target time series.
    :type orig_input:   np.ndarray
    :param cost_array:  Array of costs in optimization iterations.
    :type cost_array:   np.ndarray, optional
    :param plot_state_vars:  List of indices of state variables that should be plotted
    :type plot_state_vars:   List, optional
    :param plot_control_vars:  List of indices of control variables that should be plotted
    :type plot_control_vars:   List, optional

    """
    pass


def plot_oc_network(
    N,
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    step_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
):
    """Plot target and controlled dynamics for a network of N nodes.
    :param N:           Number of nodes in the network.
    :type N:            int
    :param duration:    Duration of simulation (in ms).
    :type duration:     float
    :param dt:          Time discretization (in ms).
    :type dt:           float
    :param state:       The state of the system controlled with the found oc-input.
    :type state:        np.ndarray
    :param target:      The target state.
    :type target:       np.ndarray
    :param control:     The control signal found by the oc-algorithm.
    :type control:      np.ndarray
    :param orig_input:  The inputs that were used to generate target time series.
    :type orig_input:   np.ndarray
    :param cost_array:  Array of costs in optimization iterations.
    :type cost_array:   np.ndarray, optional
    :param step_array:  Array of step sizes in optimization iterations.
    :type step_array:   np.ndarray, optional
    :param plot_state_vars:  List of indices of state variables that should be plotted
    :type plot_state_vars:   List, optional
    :param plot_control_vars:  List of indices of control variables that should be plotted
    :type plot_control_vars:   List, optional
    """
    pass


plt.show()
