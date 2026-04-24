import copy
import datetime
import logging
import multiprocessing
import os
import pathlib
import numpy as np
import pandas as pd
import psutil
import pypet
import tqdm
import xarray as xr
from ...utils import paths
from ...utils import pypetUtils as pu
from ...utils.collections import dotdict, flat_dict_to_nested, flatten_nested_dict, unwrap_star_dotdict

class BoxSearch:
    """
    Paremeter box search for a given model and a range of parameters.
    """

    def __init__(self, model=None, parameterSpace=None, evalFunction=None, filename=None, saveAllModelOutputs=False, ncores=None):
        """Either a model has to be passed, or an evalFunction. If an evalFunction
        is passed, then the evalFunction will be called and the model is accessible to the
        evalFunction via `self.getModelFromTraj(traj)`. The parameters of the current
        run are accessible via `self.getParametersFromTraj(traj)`.

        If no evaluation function is passed, then the model is simulated using `Model.run()`
        for every parameter.

        :param model: Model to run for each parameter (or model to pass to the evaluation function if an evaluation
            function is used), defaults to None
        :type model: `neurolib.models.model.Model`, optional
        :param parameterSpace: Parameter space to explore, defaults to None
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`, optional
        :param evalFunction: Evaluation function to call for each run., defaults to None
        :type evalFunction: function, optional
        :param filename: HDF5 storage file name, if left empty, defaults to ``exploration.hdf``
        :type filename: str
        :param saveAllModelOutputs: If True, save all outputs of model, else only default output of the model
            (and BOLD if available), defaults to False
        :type saveAllModelOutputs: bool

        :param ncores: Number of cores to simulate on (max cores default), defaults to None
        :type ncores: int, optional
        """
        self.model = model
        if evalFunction is None and model is not None:
            self.evalFunction = self._runModel
        elif evalFunction is not None:
            self.evalFunction = evalFunction
        assert evalFunction is not None or model is not None, 'Either a model has to be specified or an evalFunction.'
        assert parameterSpace is not None, 'No parameters to explore.'
        if parameterSpace.kind == 'sequence':
            assert model is not None, 'Model must be defined for sequential explore'
        self.parameterSpace = parameterSpace
        self.exploreParameters = parameterSpace.dict()
        self.useRandomICs = False
        filename = filename or 'exploration.hdf'
        self.filename = filename
        self.saveAllModelOutputs = saveAllModelOutputs
        if ncores is None:
            ncores = multiprocessing.cpu_count()
        self.ncores = ncores
        logging.info('Number of processes: {}'.format(self.ncores))
        self.initialized = False
        self._initializeExploration(self.filename)
        self.results = None

    def _initializeExploration(self, filename='exploration.hdf'):
        """Initialize the pypet environment

        :param filename: hdf filename to store the results in , defaults to "exploration.hdf"
        :type filename: str, optional
        """
        pass

    @staticmethod
    def _fillin_default_parameters_for_sequential(parametrization, model_params):
        pass

    def _addParametersToPypet(self, traj, params):
        """This function registers the parameters of the model to Pypet.
        Parameters can be nested dictionaries. They are unpacked and stored recursively.

        :param traj: Pypet trajectory to store the parameters in
        :type traj: `pypet.trajectory.Trajectory`
        :param params: Parameter dictionary
        :type params: dict, dict[dict,]
        """
        pass

    def saveToPypet(self, outputs, traj):
        """This function takes simulation results in the form of a nested dictionary
        and stores all data into the pypet hdf file.

        :param outputs: Simulation outputs as a dictionary.
        :type outputs: dict
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """
        pass

    def _runModel(self, traj):
        """If not evaluation function is given, we assume that a model will be simulated.
        This function will be called by pypet directly and therefore wants a pypet trajectory as an argument

        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """
        pass

    def _saveModelOutputsToPypet(self, traj):
        pass

    def _validatePypetParameters(self, runParams):
        """Helper to handle None's in pypet parameters
        (used for random number generator seed)

        :param runParams: parameters as returned by traj.parameters.f_to_dict()
        :type runParams: dict of pypet.parameter.Parameter
        """
        pass

    def getParametersFromTraj(self, traj):
        """Returns the parameters of the current run as a (dot.able) dictionary

        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Parameter set of the current run
        :rtype: dict
        """
        pass

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this run
        :params traj: Pypet trajectory of current run

        :returns model: Model with the parameters of this run.
        """
        pass

    def run(self, **kwargs):
        """
        Call this function to run the exploration
        """
        pass

    def loadResults(self, all=True, filename=None, trajectoryName=None, pypetShortNames=True, memory_cap=95.0):
        """Load results from a hdf file of a previous simulation.

        :param all: Load all simulated results into memory, which will be available as the `.results` attribute. Can
            use a lot of RAM if your simulation is large, please use this with caution. , defaults to True
        :type all: bool, optional
        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults
            to None
        :type trajectoryName: str, optional
        :param pypetShortNames: Use pypet short names as keys for the results dictionary. Use if you are experiencing
            errors due to natural naming collisions.
        :type pypetShortNames: bool
        :param memory_cap: Percentage memory cap between 0 and 100. If `all=True` is used, a memory cap can be set to
            avoid filling up the available RAM. Example: use `memory_cap = 95` to avoid loading more data if memory is
            at 95% use, defaults to 95
        :type memory_cap: float, int, optional
        """
        pass

    def aggregateResultsToDfResults(self, arrays=True, fillna=False):
        """Aggregate all results in to dfResults dataframe.

        :param arrays: Load array results (like timeseries) if True. If False, only load scalar results, defaults to
            True
        :type arrays: bool, optional
        :param fillna: Fill nan results (for example if they're not returned in a subset of runs) with zeros, default
            to False
        :type fillna: bool, optional
        """
        pass

    def loadDfResults(self, filename=None, trajectoryName=None):
        """Load results from a previous simulation.

        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults
            to None
        :type trajectoryName: str, optional
        """
        pass

    @staticmethod
    def _filterDictionaryBold(filt_dict, bold):
        """Filters result dictionary: either keeps ONLY BOLD results, or remove
        BOLD results.

        :param filt_dict: dictionary to filter for BOLD keys
        :type filt_dict: dict
        :param bold: whether to remove BOLD keys (bold=False) or keep only BOLD
            keys (bold=True)
        :return: filtered dict, without or only BOLD keys
        :rtype: dict
        """
        pass

    def _getCoordsFromRun(self, run_dict, bold=False):
        """Find coordinates of a single run - time, output and space dimensions.

        :param run_dict: dictionary with run results
        :type run_dict: dict
        :param bold: whether to do only BOLD or without BOLD results
        :type bold: bool
        :return: dictionary of coordinates for xarray
        :rtype: dict
        """
        pass

    def xr(self, bold=False):
        """
        Return `xr.Dataset` from the exploration results.

        :param bold: if True, will load and return only BOLD output
        :type bold: bool
        """
        pass

    def getRun(self, runId, filename=None, trajectoryName=None, pypetShortNames=True):
        """Load the simulated data of a run and its parameters from a pypetTrajectory.

        :param runId: ID of the run
        :type runId: int

        :return: Dictionary with simulated data and parameters of the run.
        :type return: dict
        """
        pass

    def getResult(self, runId):
        """Returns either a loaded result or reads from disk.

        :param runId: runId of result
        :type runId: int
        :return: result
        :rtype: dict
        """
        pass

    def info(self):
        """Print info about the current search."""
        pass