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
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    duration,
    N,
    Cmat,
    Dmat,
    K_gl,
    signalV,
    Dmat_ndt,
    ses,
    sis,
    ses_input_d,
    a_exc,
    b_exc,
    d_exc,
    tau_exc,
    gamma_exc,
    w_exc,
    exc_current,
    exc_current_baseline,
    a_inh,
    b_inh,
    d_inh,
    tau_inh,
    w_inh,
    inh_current,
    inh_current_baseline,
    J_NMDA,
    J_I,
    w_ee,
    r_exc,
    r_inh,
    noise_se,
    noise_si,
    exc_ou,
    inh_ou,
    exc_ou_mean,
    inh_ou_mean,
    tau_ou,
    sigma_ou,
):
    """
    Wong-Wang model equations (Deco2014, no long-rage feedforward inhibition):

    currents
    I_e = w_e * I_0 + w_ee * J_NMDA * s_e - J_I * s_i + K * J_NMDA * \sum Gij * s_e_j + I_ext
    I_i = w_i * I_0 + J_NMDA * s_e - s_i

    synaptic activity
    d_se/dt = (-s / tau) + (1.0 - s) * gamma * r + noise
    d_si/dt = (-s / tau) + r + noise

    firing rate transfer function
    r = (a * I - b) / (1.0 - exp(-d * (a * I - b)))

    """
    pass
