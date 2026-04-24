import logging
import numpy as np
from chspy import join
from ...utils.collections import dotdict, flat_dict_to_nested, flatten_nested_dict, star_dotdict
from ..model import Model
from .builder.base.constants import NETWORK_CONNECTIVITY, NETWORK_DELAYS
from .builder.base.network import Network, Node
DEFAULT_RUN_PARAMS = {'duration': 2000, 'dt': 0.1, 'seed': None, 'backend': 'jitcdde'}

class MultiModel(Model):
    """
    Base for all MultiModels i.e. heterogeneous networks or network nodes built
    using model builder.
    """

    @classmethod
    def init_node(cls, node):
        """
        Init model class from node.

        :param node: initialised network node from MultiModel builder
        :type node: `neurolib.models.multimodel.builder.base.network.Node`
        """
        pass

    def __init__(self, model_instance):
        assert isinstance(model_instance, (Node, Network))
        assert model_instance.initialised
        self.model_instance = model_instance
        self.name = self.model_instance.label
        self.state_vars = self.model_instance.state_variable_names
        self.default_output = self.model_instance.default_output
        self.output_vars = self.model_instance.output_vars
        assert isinstance(self.default_output, str), '`default_output` must be a string.'
        self.params = self._set_model_params()
        self.integration = None
        self.init_vars = None
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()
        self.boldInitialized = False
        self.params['sampling_dt'] = self.params['sampling_dt'] or self.params['dt']
        logging.info(f'{self.name}: Model initialized.')

    def _set_model_params(self):
        """
        Set all necessary model parameters.
        """
        pass

    def _sync_model_params(self):
        """
        Pulls params from `model_instance` and updates self.params.
        """
        pass

    def getMaxDelay(self):
        """
        Return max delay in units of dt. In ms, this is given as a property in the model instance.
        """
        pass

    def _update_model_params(self):
        pass

    @property
    def num_noise_variables(self):
        pass

    @property
    def num_state_variables(self):
        pass

    @property
    def noise_input(self):
        pass

    @noise_input.setter
    def noise_input(self, new_noise):
        pass

    def run(self, chunkwise=False, chunksize=None, bold=False, append_outputs=False, continue_run=False, noise_input=None):
        pass

    def _init_noise_inputs(self, backend):
        """
        Build noise / stimulus input to the model.
        """
        pass

    def integrate(self, append_outputs=False, simulate_bold=False, noise_input=None):
        """
        :param noise_input: custom noise input if desired, if None, will use
            default, it's type depends on backend:
            - for `numba` backend as np.ndarray
            - for `jitcdde` backend as interpolated Cubic Hermite Splines
                (`chspy.CubicHermiteSpline`)
        :type noise_input: np.ndarray|chspy.CubicHermiteSpline
        """
        pass

    def setInitialValuesToLastState(self):
        pass

    def integrateChunkwise(self, chunksize, bold, append_outputs):
        pass

    def storeOutputsAndStates(self, results, append):
        pass