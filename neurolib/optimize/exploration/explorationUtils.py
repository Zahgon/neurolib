import os
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

import logging
import tqdm

from scipy import stats

from ...utils import functions as func
from ...utils import paths as paths


def plotExplorationResults(
    dfResults,
    par1,
    par2,
    plot_key,
    nan_to_zero=False,
    by=None,
    by_label=None,
    plot_key_label=None,
    symmetric_colorbar=False,
    one_figure=False,
    contour=None,
    alpha_mask=None,
    multiply_axis=None,
    savename=None,
    **kwargs,
):
    """ """
    pass


def contourPlotDf(
    dataframe,
    color="white",
    levels=None,
    ax=None,
    alpha=1.0,
    countour=True,
    contourf=False,
    clabel=False,
    **contour_kwargs,
):
    pass


def alphaMask(image, threshold, alpha, mask=None, invert=False, style=None):
    """Create an alpha mask on an image using a threshold

    :param image: RGB image to create a mask on.
    :type image: np.array (NxNx3)
    :param threshold: Threshold value
    :type threshold: float
    :param alpha: Alpha value of mask
    :type alpha: float
    :param mask: A predefined mask that can be used instead of the image itself, defaults to None
    :type mask: np.array, optional
    :param invert: Invert the mask, defaults to False
    :type invert: bool, optional
    :param style: Chose a style for the mask, currently only `stripes` supported, defaults to None
    :type style: string, optional
    :return: Masked image (RGBA), 4-dimensional (NxNx4)
    :rtype: np.array
    """
    pass


def plotResult(search, runId, z_bold=False, **kwargs):
    pass


def processExplorationResults(search, **kwargs):
    """Process results from the exploration."""
    pass


def computeMinMax(dfResults, i, output, output_name):
    # calculate the maximum of the output
    pass


def findCloseResults(dfResults, dist=None, relative=False, **kwargs):
    """Filter and get a list of results from a pandas dataframe that are close to the variables specified here.

    Use the parameters to filter for as kwargs:
    Usage: findCloseResults(search.dfResults, mue_ext_mean=2.0, mui_ext_mean=2.5)

    Alternatively, use ranges a la [min, max] for each parameter.
    Usage: findCloseResults(search.dfResults, mue_ext_mean=[2.0, 3.0], mui_ext_mean=2.5)

    :param dfResults: Pandas dataframe to filter
    :type dfResults: pandas.DataFrame
    :param dist: Distance to specified points in kwargs, defaults to None
    :type dist: float, optional
    :param relative: Relative distance (percentage) or absolute distance, defaults to False
    :type relative: bool, optional
    :return: Filtered Pandas dataframe
    :rtype: pandas.DataFrame
    """
    pass


def paramsRun(dfResults, runNr):
    pass
