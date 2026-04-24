import logging
import xarray as xr
import numpy as np
from ..models import bold
from ..utils.collections import dotdict

class Model:
    """The Model base class runs models, manages their outputs, parameters and more.
    This class should serve as the base class for all implemented models.
    """

    def __init__(self, integration, params):
        if hasattr(self, 'name'):
            if self.name is not None:
                assert isinstance(self.name, str), f'Model name is not a string.'
        else:
            self.name = 'Noname'
        assert integration is not None, 'Model integration function not given.'
        self.integration = integration
        assert isinstance(params, dict), 'Parameters must be a dictionary.'
        self.params = dotdict(params)
        assert hasattr(self, 'state_vars'), f'Model {self.name} has no attribute `state_vars`, which should be alist of strings containing all variable names.'
        assert np.all([type(s) is str for s in self.state_vars]), 'All entries in state_vars must be strings.'
        assert hasattr(self, 'default_output'), f'Model {self.name} needs to define a default output variable in `default_output`.'
        assert isinstance(self.default_output, str), '`default_output` must be a string.'
        if not hasattr(self, 'output_vars'):
            self.output_vars = self.state_vars
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()
        self.boldInitialized = False
        logging.info(f'{self.name}: Model initialized.')

    def initializeBold(self):
        """Initialize BOLD model."""
        pass

    def get_bold_variable(self, variables):
        pass

    def simulateBold(self, bold_variable, append=True):
        """Gets the default output of the model and simulates the BOLD model.
        Adds the simulated BOLD signal to outputs.
        """
        pass

    def checkChunkwise(self, chunksize):
        """Checks if the model fulfills requirements for chunkwise simulation.
        Checks whether the sampling rate for outputs fits to chunksize and duration.
        Throws errors if not."""
        pass

    def setSamplingDt(self):
        """Checks if sampling_dt is set correctly and sets self.`sample_every`
        1) Check if sampling_dt is multiple of dt
        2) Check if semplind_dt is greater than duration
        """
        pass

    def initializeRun(self, initializeBold=False):
        """Initialization before each run.

        :param initializeBold: initialize BOLD model
        :type initializeBold: bool
        """
        pass

    def run(self, chunkwise=False, chunksize=None, bold=False, append_outputs=False, continue_run=False):
        """
        Main interfacing function to run a model.

        The model can be run in three different ways:
        1) `model.run()` starts a new run.
        2) `model.run(chunkwise=True)` runs the simulation in chunks of length `chunksize`.
        3) `mode.run(continue_run=True)` continues the simulation of a previous run. This has no effect during the first run.

        :param inputs: list of inputs to the model, must have the same order as model.input_vars. Note: no sanity check is performed for performance reasons. Take care of the inputs yourself.
        :type inputs: list[np.ndarray|]
        :param chunkwise: simulate model chunkwise or in one single run, defaults to False
        :type chunkwise: bool, optional
        :param chunksize: size of the chunk to simulate in dt, if set will imply chunkwise=True, defaults to 2s
        :type chunksize: int, optional
        :param bold: simulate BOLD signal (only for chunkwise integration), defaults to False
        :type bold: bool, optional
        :param append_outputs: append new and chunkwise outputs to the outputs attribute, defaults to False. Note: BOLD outputs are always appended.
        :type append_outputs: bool, optional
        :param continue_run: continue a simulation by using the initial values from a previous simulation. This has no effect during the first run.
        :type continue_run: bool
        """
        pass

    def checkOutputs(self):
        pass

    def integrate(self, append_outputs=False, simulate_bold=False):
        """Calls each models `integration` function and saves the state and the outputs of the model.

        :param append: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append: bool, optional
        """
        pass

    def integrateChunkwise(self, chunksize, bold=False, append_outputs=False):
        """Repeatedly calls the chunkwise integration for the whole duration of the simulation.
        If `bold==True`, the BOLD model is simulated after each chunk.

        :param chunksize: size of each chunk to simulate in units of dt
        :type chunksize: int
        :param bold: simulate BOLD model after each chunk, defaults to False
        :type bold: bool, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append_outputs: bool, optional
        """
        pass

    def clearModelState(self):
        """Clears the model's state to create a fresh one"""
        pass

    def storeOutputsAndStates(self, t, variables, append=False):
        """Takes the simulated variables of the integration and stores it to the appropriate model output and state object.

        :param t: time vector
        :type t: list
        :param variables: variable from time integration
        :type variables: numpy.ndarray
        :param append: append output to existing output or overwrite, defaults to False
        :type append: bool, optional
        """
        pass

    def setInitialValuesToLastState(self):
        """Reads the last state of the model and sets the initial conditions to that state for continuing a simulation."""
        pass

    def randomICs(self, min=0, max=1):
        """Generates a new set of uniformly-distributed random initial conditions for the model.

        TODO: All parameters are drawn from the same distribution / range. Allow for independent ranges.

        :param min: Minium of uniform distribution
        :type min: float
        :param max: Maximum of uniform distribution
        :type max: float
        """
        pass

    def setInputs(self, inputs):
        """Take inputs from a list and store it in the appropriate model parameter for external input.
        TODO: This is not safe yet, checks should be implemented whether the model has inputs defined or not for example.

        :param inputs: list of inputs
        :type inputs: list[np.ndarray(), ...]
        """
        pass

    def autochunk(self, inputs=None, chunksize=1, append_outputs=False, bold=False):
        """Executes a single chunk of integration, either for a given duration
        or a single timestep `dt`. Gathers all inputs to the model and resets
        the initial conditions as a preparation for the next chunk.

        :param inputs: list of input values, ordered according to self.input_vars, defaults to None
        :type inputs: list[np.ndarray|], optional
        :param chunksize: length of a chunk to simulate in dt, defaults 1
        :type chunksize: int, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append_outputs: bool, optional
        """
        pass

    def getMaxDelay(self):
        """Computes the maximum delay of the model. This function should be overloaded
        if the model has internal delays (additional to delay between nodes defined by Dmat)
        such as the delay between an excitatory and inhibitory population within each brain area.
        If this function is not overloaded, the maximum delay is assumed to be defined from the
        global delay matrix `Dmat`.

        Note: Maxmimum delay is given in units of dt.

        :return: maxmimum delay of the model in units of dt
        :rtype: int
        """
        pass

    def setStateVariables(self, name, data):
        """Saves the models current state variables.

        TODO: Cut state variables to length of self.maxDelay
        However, this could be time-memory tradeoff

        :param name: name of the state variable
        :type name: str
        :param data: value of the variable
        :type data: np.ndarray
        """
        pass

    def setOutput(self, name, data, append=False, removeICs=False):
        """Adds an output to the model, typically a simulation result.
        :params name: Name of the output in dot.notation, a la "outputgroup.output"
        :type name: str
        :params data: Output data, can't be a dictionary!
        :type data: `numpy.ndarray`
        """
        pass

    def getOutput(self, name):
        """Get an output of a given name (dot.semarated)
        :param name: A key, grouped outputs in the form group.subgroup.variable
        :type name: str

        :returns: Output data
        """
        pass

    def __getitem__(self, key):
        """Index outputs with a dictionary-like key, e.g., `model['rates_exc']`."""
        return self.getOutput(key)

    def getOutputs(self, group=''):
        """Get all outputs of an output group. Examples: `getOutputs("BOLD")` or simply `getOutputs()`

        :param group: Group name, subgroups separated by dots. If left empty (default), all outputs of the root group
            are returned.
        :type group: str
        """
        pass

    @property
    def output(self):
        """Returns value of default output as defined by `self.default_output`.
        Note that all outputs are saved in the attribute `self.outputs`.
        """
        pass

    def xr(self, group=''):
        """Converts a group of outputs to xarray. Output group needs to contain an
        element that starts with the letter "t" or it will not recognize any time axis.

        :param group: Output group name, example:  "BOLD". Leave empty for top group.
        :type group: str
        """
        pass