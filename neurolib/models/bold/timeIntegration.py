import numpy as np
import numba

def simulateBOLD(Z, dt, voxelCounts=None, X=None, F=None, Q=None, V=None):
    """Simulate BOLD activity using the Balloon-Windkessel model.
    See Friston 2000, Friston 2003 and Deco 2013 for reference on how the BOLD signal is simulated.
    The returned BOLD signal should be downsampled to be comparable to a recorded fMRI signal.

    :param Z: Synaptic activity
    :type Z: numpy.ndarray
    :param dt: dt of input activity in s
    :type dt: float
    :param voxelCounts: Number of voxels in each region (not used yet!) # TODO
    :type voxelCounts: numpy.ndarray
    :param X: Initial values of Vasodilatory signal, defaults to None
    :type X: numpy.ndarray, optional
    :param F: Initial values of Blood flow, defaults to None
    :type F: numpy.ndarray, optional
    :param Q: Initial values of Deoxyhemoglobin, defaults to None
    :type Q: numpy.ndarray, optional
    :param V: Initial values of Blood volume, defaults to None
    :type V: numpy.ndarray, optional

    :return: BOLD, X, F, Q, V
    :rtype: (numpy.ndarray,)
    """
    pass

@numba.njit
def integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau):
    """Integrate the Balloon-Windkessel model.

    Reference:

    Friston et al. (2000), Nonlinear responses in fMRI: The balloon model, Volterra kernels, and other hemodynamics.
    Friston et al. (2003), Dynamic causal modeling

    Variable names in Friston2000:
    X = x1, Q = x4, V = x3, F = x2

    Friston2003: see Equation (3)

    NOTE: A very small constant EPS is added to F to avoid F become too small / negative
    and cause a floating point error in EQ. Q due to the exponent **(1 / F[j])

    """
    pass