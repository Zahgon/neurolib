import logging
import numpy as np
import scipy.signal
import numba

"""Collection of useful functions for data processing.
"""


def kuramoto(traces, smoothing=0.0, distance=10, prominence=5):
    """
    Computes the Kuramoto order parameter of a timeseries which is a measure for synchrony.
    Can smooth timeseries if there is noise.
    Peaks are then detected using a peakfinder. From these peaks a phase is derived and then
    the amount of phase synchrony (the Kuramoto order parameter) is computed.

    :param traces: Multidimensional timeseries array
    :type traces: numpy.ndarray
    :param smoothing: Gaussian smoothing strength
    :type smoothing: float, optional
    :param distance: minimum distance between peaks in samples
    :type distance: int, optional
    :param prominence: vertical distance between the peak and its lowest contour line
    :type prominence: int, optional

    :return: Timeseries of Kuramoto order paramter
    :rtype: numpy.ndarray
    """
    pass


def matrix_correlation(M1, M2):
    """Pearson correlation of the lower triagonal of two matrices.
    The triangular matrix is offset by k = 1 in order to ignore the diagonal line

    :param M1: First matrix
    :type M1: numpy.ndarray
    :param M2: Second matrix
    :type M2: numpy.ndarray
    :return: Correlation coefficient
    :rtype: float
    """
    pass


def weighted_correlation(x, y, w):
    """Weighted Pearson correlation of two series.

    :param x: Timeseries 1
    :type x: list, np.array
    :param y: Timeseries 2, must have same length as x
    :type y: list, np.array
    :param w: Weight vector, must have same length as x and y
    :type w: list, np.array
    :return: Weighted correlation coefficient
    :rtype: float
    """
    pass


def fc(ts):
    """Functional connectivity matrix of timeseries multidimensional `ts` (Nxt).
    Pearson correlation (from `np.corrcoef()` is used).

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :return: N x N functional connectivity matrix
    :rtype: numpy.ndarray
    """
    pass


def fcd(ts, windowsize=30, stepsize=5):
    """Computes FCD (functional connectivity dynamics) matrix, as described in Deco's whole-brain model papers.
    Default paramters are suited for computing FCS matrices of BOLD timeseries:
    A windowsize of 30 at the BOLD sampling rate of 0.5 Hz equals 60s and stepsize = 5 equals 10s.

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :param windowsize: Size of each rolling window in timesteps, defaults to 30
    :type windowsize: int, optional
    :param stepsize: Stepsize between each rolling window, defaults to 5
    :type stepsize: int, optional
    :return: T x T FCD matrix
    :rtype: numpy.ndarray
    """
    pass


def matrix_kolmogorov(m1, m2):
    """Computes the Kolmogorov distance between the distributions of lower-triangular entries of two matrices
    See: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test

    :param m1: matrix 1
    :type m1: np.ndarray
    :param m2: matrix 2
    :type m2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    pass


def ts_kolmogorov(ts1, ts2, **fcd_kwargs):
    """Computes kolmogorov distance between two timeseries.
    This is done by first computing two FCD matrices (one for each timeseries)
    and then measuring the Kolmogorov distance of the upper triangle of these matrices.

    :param ts1: Timeseries 1
    :type ts1: np.ndarray
    :param ts2: Timeseries 2
    :type ts2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    pass


# def max_distance_cumulative(data1, data2):
#     """
#     From: https://github.com/scipy/scipy/issues/9389

#     Computes the maximal vertical distance between cumulative distributions
#     (this is the statistic for KS tests). Code mostly copied from
#     scipy.stats.ks_twosamp

#     Parameters
#     ----------
#     data1 : array_like
#         First data set
#     data2 : array_like
#         Second data set
#     Returns
#     -------
#     d : float
#         Max distance, i.e. value of the Kolmogorov Smirnov test. Sign is + if
#         the cumulative of data1 < the one of data2 at that location, else -.
#     x : float
#         Value of x where maximal distance d is reached.
#     """
#     from numpy import ma

#     (data1, data2) = (ma.asarray(data1), ma.asarray(data2))
#     (n1, n2) = (data1.count(), data2.count())
#     mix = ma.concatenate((data1.compressed(), data2.compressed()))
#     mixsort = mix.argsort(kind="mergesort")
#     csum = np.where(mixsort < n1, 1.0 / n1, -1.0 / n2).cumsum()

#     # Check for ties
#     if len(np.unique(mix)) < (n1 + n2):
#         ind = np.r_[np.diff(mix[mixsort]).nonzero()[0], -1]
#         csum = csum[ind]
#         mixsort = mixsort[ind]

#     csumabs = ma.abs(csum)
#     i = csumabs.argmax()

#     d = csum[i]
#     # mixsort[i] contains the index of mix with the max distance
#     x = mix[mixsort[i]]

#     return (d, x)


# def print_params(params):
#     """
#     Helpfer function for printing a subset of the paramters of the aln model.
#     Todo: This function should not be here, it is too specific for the aln model.
#     Idea: A model could register "parameters of interest" and be printed with this function.
#     However, this should be placed in the Model class in any case
#     """
#     paramsOfInterest = [
#         "dt",
#         "Ke_gl",
#         "mue_ext_mean",
#         "mui_ext_mean",
#         "sigma_ou",
#         "signalV",
#         "a",
#         "b",
#         "Jee_max",
#         "Jie_max",
#         "Jii_max",
#         "Jei_max",
#         "cee",
#         "cie",
#         "cii",
#         "cei",
#         "Ke",
#         "Ki",
#         "de",
#         "di",
#     ]
#     for p in paramsOfInterest:
#         print("params['%s'] = %0.3f" % (p, params[p]))


def getPowerSpectrum(activity, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns a power spectrum using Welch's method.

    :param activity: One-dimensional timeseries
    :type activity: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """
    pass


def getMeanPowerSpectrum(activities, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns the mean power spectrum of multiple timeseries.

    :param activities: N-dimensional timeseries
    :type activities: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """
    pass
