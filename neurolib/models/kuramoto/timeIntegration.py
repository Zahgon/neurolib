import numpy as np
import numba

from ...utils import model_utils as mu


def timeIntegration(params):
    """
    setting up parameters for time integration
    
    :param params: Parameter dictionary of the model
    :type params: dict

    :return: Integrated activity of the model
    :rtype: (numpy.ndarray, )
    """ 
    pass


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t, 
    dt, 
    sqrt_dt,
    N,
    omega,
    k_n, 
    Cmat,
    Dmat,
    theta,
    theta_ext,
    tau_ou,
    sigma_ou,
    theta_ou,
):
    """
    Kuramoto Model 
    """
    pass
