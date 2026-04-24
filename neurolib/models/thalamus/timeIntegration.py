import numba
import numpy as np


def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """
    pass


@numba.njit()
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    Q_max,
    C1,
    theta,
    sigma,
    g_L,
    E_L,
    g_AMPA,
    g_GABA,
    E_AMPA,
    E_GABA,
    g_LK,
    E_K,
    g_T_t,
    g_T_r,
    E_Ca,
    g_h,
    g_inc,
    E_h,
    C_m,
    tau,
    alpha_Ca,
    Ca_0,
    tau_Ca,
    k1,
    k2,
    k3,
    k4,
    n_P,
    gamma_e,
    gamma_r,
    d_phi,
    noise,
    ext_current_t,
    ext_current_r,
    N_rt,
    N_tr,
    N_rr,
    V_t,
    V_r,
    Q_t,
    Q_r,
    Ca,
    h_T_t,
    h_T_r,
    m_h1,
    m_h2,
    s_et,
    s_gt,
    s_er,
    s_gr,
    ds_et,
    ds_gt,
    ds_er,
    ds_gr,
):
    pass
