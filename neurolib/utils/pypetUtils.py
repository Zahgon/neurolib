import h5py
import pypet
import pathlib
import logging
import copy

from .collections import dotdict


def getTrajectorynamesInFile(filename):
    """
    Return a list of all pypet trajectory names in a a given hdf5 file.

    :param filename:  Name of the hdf file
    :type filename: str

    :return: List of strings containing the trajectory names
    :rtype: list[str]
    """
    pass


def loadPypetTrajectory(filename, trajectoryName):
    """Read HDF file with simulation results and return the chosen trajectory.

    :param filename: HDF file path
    :type filename: str

    :return: pypet trajectory
    """
    pass


def getRun(runId, pypetTrajectory, pypetShortNames=True):
    """Load the simulated data of a run and its parameters from a pypetTrajectory.

    :param runId: ID of the run
    :type runId: int
    :param pypetTrajectory: Pypet trajectory to get run from.
    :type pypetTrajectory: pypet.Trajectory
    :param pypetShortNames: Use pypet short names as keys for the results dictionary. Use if you are experiencing errors due to natural naming collisions.
    :type pypetShortNames: bool

    :return: Dictionary with simulated data and parameters of the run.
    :type return: dict
    """
    pass
