import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(seed=None):
    """
    Load default parameters for the thalamic mass model due to Costa et al.
    Subscript t (_t) referes to thalamocortical relay population (TCR), while
    sucscript r (_r) referes to the thalamic reticular nuclei (TRN).

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """
    pass


def generateRandomICs(seed=None):
    """Generates random Initial Conditions for the interareal network

    :returns:   A tuple of 15 floats for representing initial state of the
                thalamus
    """
    pass
