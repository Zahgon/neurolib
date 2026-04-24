import logging
import os
import numpy as np
import pandas as pd
from ...utils import paths as paths
from . import deapUtils as du

def saveToPypet(traj, pop, gIdx):
    pass

def printParamDist(pop=None, paramInterval=None, gIdx=None):
    pass

def printIndividuals(pop, paramInterval, stats=True):
    pass

def plotScoresDistribution(scores, gIdx, save_plots=None, color='C0'):
    pass

def plotSeabornScatter1(evolution, dfPop=None, save_plots=None, color='C0', line_kws={}, scatter_kws={}):
    pass

def plotSeabornScatter2(evolution, dfPop=None, save_plots=None, color='C0'):
    pass

def plotPopulation(evolution, history=False, plotDistribution=True, plotScattermatrix=False, save_plots=None, color='C0'):
    """
    Print some stats of a population fitness
    """
    pass

def plotProgress(evolution, reverse=True):
    pass

def printEvolutionInfo(evolution):
    """Function that prints all important parameters of the evolution.
    :param evolution: evolution object
    """
    pass