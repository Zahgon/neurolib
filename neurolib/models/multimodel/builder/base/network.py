import logging
from copy import deepcopy
from itertools import chain, islice
import numpy as np
import symengine as se
import sympy as sp
from jitcdde import t as time_vector
from jitcdde import y as state_vector
from .....utils.collections import flat_dict_to_nested, flatten_nested_dict
from .backend import BackendIntegrator
from .constants import EXC, NETWORK_CONNECTIVITY, NETWORK_DELAYS, NODE_CONNECTIVITY, NODE_DELAYS
from .neural_mass import NeuralMass
from .params import float_params_to_individual_symbolic, float_params_to_vector_symbolic

def _sanitize_matrix(matrix, target_shape):
    """
    Sanitize matrix before assigning to connectivity or delay - check shape
    and cast to float if necessary.

    :param matrix: input matrix to sanitize
    :type matrix: np.ndarray|sp.MatrixSymbol
    :param target_shape: shape of the matrix
    :type target_shape: tuple
    :return: sanitized matrix
    :rtype: np.ndarray|sp.MatrixSymbol
    """
    pass

class Node(BackendIntegrator):
    """
    Base class for all nodes within the network.
    """
    name = ''
    label = ''
    index = None
    sync_variables = []
    default_network_coupling = {}
    default_output = None
    output_vars = []

    def __init__(self, neural_masses):
        """
        :param neural_masses: list of neural masses in this node
        :type neural_masses: list[NeuralMass]
        """
        assert all((isinstance(mass, NeuralMass) for mass in neural_masses))
        self.masses = neural_masses
        self.num_state_variables = sum([mass.num_state_variables for mass in self])
        self.num_noise_variables = sum([mass.num_noise_variables for mass in self])
        assert len(self.noise_input) == self.num_noise_variables
        self.idx_state_var = None
        self.initialised = False
        assert self.default_output in self.state_variable_names[0]
        assert all((var in self.state_variable_names[0] for var in self.output_vars))
        mass_types = [mass.mass_type for mass in self]
        assert len(set(mass_types)) == len(mass_types), f'Mass types needs to be different: {mass_types}'
        self._initial_state = None
        self.float_params = None
        self.are_params_floats = True

    def __str__(self):
        """
        String representation.
        """
        return f"Network node: {self.name} with {len(self.masses)} neural mass(es): {', '.join([mass.name for mass in self])}"

    def __repr__(self):
        return self.__str__()

    def describe(self):
        """
        Return description dict.
        """
        pass

    def __len__(self):
        """
        Get length.
        """
        return len(self.masses)

    def __getitem__(self, index):
        """
        Get item.
        """
        return self.masses[index]

    @property
    def state_variable_names(self):
        pass

    @property
    def max_delay(self):
        pass

    @property
    def noise_input(self):
        pass

    @noise_input.setter
    def noise_input(self, new_noise):
        pass

    def get_nested_params(self):
        """
        Return nested dictionary with parameters from all masses within this
        node.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        pass

    def make_params_symbolic(self, vector=True):
        """
        Make all node parameters symbolic, instead of concrete values. Useful
        when caching compiled functions.

        :param vector: create vectorised params
        :type vector: bool
        """
        pass

    def make_params_floats(self):
        """
        Make all node parameters floats again!
        """
        pass

    def init_node(self, **kwargs):
        """
        Initialise node and all the masses within.

        :kwargs: optional keyword arguments to init_mass
        """
        pass

    def _sanitize_update_params(self, params_dict):
        """
        If dictionary with parameters for update have one title level, trim this.
        """
        pass

    def update_params(self, params_dict, **kwargs):
        """
        Update parameters of the node, i.e. recursively update all parameters of masses within this node.

        :param params_dict: new parameters for this node, same format as
            `get_nested_params`, i.e. nested dict
        :type params_dict: dict
        """
        pass

    @staticmethod
    def _get_index(symbol_name):
        """
        Gets index value from the symbol name.
        """
        pass

    @staticmethod
    def _strip_index(symbol_name):
        """
        Strip index value from the symbol name.
        """
        pass

    def _callbacks(self):
        """
        Gather callbacks from all masses.
        """
        pass

    def _numba_callbacks(self):
        """
        Gather callbacks from all masses.
        """
        pass

    def _sync(self):
        """
        Define synchronisation step for the node. This should define the helper
        and its equations. Must be symbolic, i.e. defined with basic math
        operators and symengine operations on state vector. Should return list
        as
        [(se.Symbol, <symbolic definition>)]
        """
        pass

    @property
    def initial_state(self):
        """
        Return initial state of this node, i.e. sum of initial states of all masses.
        """
        pass

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        Manually set initial state - be sure what you are doing!

        :param initial_state: vector representing the initial state, if 2D pass as nodes x time
        :type initial_state: np.ndarray
        """
        pass

    def all_couplings(self, mass_indices=None):
        """
        Return coupling variable names of all masses within this node, denoted by each masses index.

        :param mass_indices: indices of masses
        :type mass_indices: list|None
        :return: coupling variables from all masses indexed by mass_indices
        :rtype: dict
        """
        pass

    def _derivatives(self, network_coupling=None):
        """
        Gather all derivatives from all masses.

        :param network_coupling: dict of network coupling for this node, if
            None, default network coupling will be used
        :type network_coupling: dict|None
        :return: derivatives of the state vector for this node
        :rtype: list
        """
        pass

class SingleCouplingExcitatoryInhibitoryNode(Node):
    """
    Basic node with arbitrary number of excitatory and inhibitory populations,
    but the coupling is through one variable - usually firing rate of the
    population. Will compute connectivity within node as per mass types.
    """
    name = 'Single coupling excitatory vs inhibitory node'
    label = '1ExcInhNode'
    sync_variables = ['node_exc_exc', 'node_inh_exc', 'node_exc_inh', 'node_inh_inh']
    default_network_coupling = {'network_exc_exc': 0.0}

    def __init__(self, neural_masses, local_connectivity, local_delays=None):
        """
        :param neural_masses: list of neural masses in this node
        :type neural_masses: list[NeuralMass]
        :param local_connectivity: connectivity matrix for within node
            connections - same order as neural masses, matrix as [to, from]
        :type local_connectivity: np.ndarray
        :param local_delays: delay matrix for within node connections - same
            order as neural masses, if None, delays are all zeros, in ms,
            matrix as [to, from]
        :type local_delays: np.ndarray|None
        """
        super().__init__(neural_masses=neural_masses)
        assert all((len(mass.coupling_variables) == 1 for mass in self.masses))
        assert local_connectivity.shape[0] == len(self.masses)
        if local_delays is None:
            local_delays = np.zeros_like(local_connectivity)
        assert local_connectivity.shape == local_delays.shape
        self.connectivity = local_connectivity
        self.delays = local_delays
        self.excitatory_masses = np.array([mass.mass_type == EXC for mass in self.masses])
        self.inhibitory_masses = ~self.excitatory_masses
        self.excitatory_masses = np.where(self.excitatory_masses)[0]
        self.inhibitory_masses = np.where(self.inhibitory_masses)[0]

    def __str__(self):
        """
        String representation.
        """
        mass_names = ', '.join([f'{mass.name} {mass.mass_type}' for mass in self.masses])
        return f'Network node: {self.name} with {len(self.masses)} neural masses: {mass_names}'

    def describe(self):
        """
        Return description dict.
        """
        pass

    @property
    def max_delay(self):
        pass

    def get_nested_params(self):
        """
        Add local connectivity and local delays matrices to mass parameters.

        :return: nested parameters containing also connectivity
        :rtype: dict
        """
        pass

    def init_node(self, **kwargs):
        """
        Init node and construct input matrix as [to, from] array of state
        vectors with correct delays.
        """
        pass

    def update_params(self, params_dict, **kwargs):
        """
        Update params - also update local connectivity and local delays,
        then pass to base class.
        """
        pass

    def _sync(self):
        pass

class Network(BackendIntegrator):
    """
    Base class for brain network.
    """
    name = ''
    label = ''
    sync_variables = []
    default_coupling = {}
    default_output = None
    output_vars = []

    def __init__(self, nodes, connectivity_matrix, delay_matrix=None):
        """
        :param nodes: list of nodes in this network
        :type nodes: list[Node]
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [to,
            from]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [to, from]
        :type delay_matrix: np.ndarray|None
        """
        self.nodes = nodes
        self.num_state_variables = sum([node.num_state_variables for node in self])
        self.num_noise_variables = sum([node.num_noise_variables for node in self])
        assert len(self.noise_input) == self.num_noise_variables
        assert connectivity_matrix.shape[0] == self.num_nodes
        if delay_matrix is None:
            delay_matrix = np.zeros_like(connectivity_matrix)
        assert connectivity_matrix.shape == delay_matrix.shape
        self.connectivity = connectivity_matrix
        self.delays = delay_matrix
        self._initial_state = None
        self.initialised = False
        if self.default_output is None:
            default_output = set([node.default_output for node in self])
            assert len(default_output) == 1
            self.default_output = next(iter(default_output))
        assert all((var in chain.from_iterable(self.state_variable_names) for var in self.output_vars))
        assert all((self.default_output in node_state_vars for node_state_vars in self.state_variable_names))
        self.float_params = None
        self.are_params_floats = True
        self.init_network()

    def __str__(self):
        """
        String representation.
        """
        return f'Brain network {self.name} with {self.num_nodes} nodes'

    def __repr__(self):
        return self.__str__()

    def describe(self):
        """
        Return description dict.
        """
        pass

    def __len__(self):
        """
        Get length.
        """
        return len(self.nodes)

    def __getitem__(self, index):
        """
        Get item.
        """
        return self.nodes[index]

    @property
    def max_delay(self):
        pass

    @property
    def num_nodes(self):
        pass

    @property
    def state_variable_names(self):
        pass

    @property
    def initial_state(self):
        """
        Return initial state for whole network.
        """
        pass

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        Manually set initial state - be sure what you are doing!

        :param initial_state: vector representing the initial state, if 2D pass as nodes x time
        :type initial_state: np.ndarray
        """
        pass

    @property
    def noise_input(self):
        pass

    @noise_input.setter
    def noise_input(self, new_noise):
        pass

    @staticmethod
    def _strip_index(symbol_name):
        """
        Strip index value from the symbol name.
        """
        pass

    @staticmethod
    def _strip_node_idx(symbol_name):
        """
        Keep index value from the symbol name.
        """
        pass

    @staticmethod
    def _prepare_mass_params(param, num_nodes, native_type=dict):
        """
        Prepare mass parameters for the network. I.e. extend to list of the
        same length as number of nodes.

        :param param: parameters to check / prepare
        :type param: list[native_type]|native_type|None
        :param num_nodes: number of nodes in the network
        :type num_nodes: int
        """
        pass

    def get_nested_params(self):
        """
        Return nested dictionary with parameters from all nodes and all masses
        within this network.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        pass

    def make_params_symbolic(self, vector=True):
        """
        Make all node parameters symbolic, instead of concrete values. Useful
        when caching compiled functions.

        :param vector: create vectorised params
        :type vector: bool
        """
        pass

    def make_params_floats(self):
        """
        Make all node parameters floats again!
        """
        pass

    def init_network(self, **kwargs):
        """
        Initialise network and the nodes within.

        :kwargs: optional keyword arguments to init_node
        """
        pass

    def update_params(self, params_dict, **kwargs):
        """
        Update parameters of this network, i.e. recursively for all nodes and
        all masses.

        :param params_dict: new parameters for the network
        :type params_dict: dict
        """
        pass

    def _callbacks(self):
        """
        Gather callbacks from all nodes.
        """
        pass

    def _numba_callbacks(self):
        """
        Gather callbacks from all nodes.
        """
        pass

    def _sync(self):
        """
        Define synchronisation step for whole network. This should define helper
        and its equations. Must be symbolic, i.e. defined with basic math
        operators and symengine operations on state vector. Should return list
        as
        [(se.Symbol, <symbolic definition>)]
        """
        pass

    def _construct_input_matrix(self, within_node_idx):
        """
        Construct input matrix as [to, from] with correct state variables and
        delays.

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :return: matrix of delayed inputs as [to, from]
        :rtype: np.ndarray
        """
        pass

    def _no_coupling(self, symbol):
        """
        Turn off coupling for given symbol.

        :param symbol: which symbol to fill with the data
        :type symbol: str
        """
        pass

    def _diffusive_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform diffusive coupling on the network with given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx] * (X[idx](t - Dmat) - X[to](t)))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity: connectivity matrix - if None will use the network
            connectivity from init
        :type connectivity: np.ndarray
        """
        pass

    def _multiplicative_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform multiplicative coupling on the network with given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx]*X[idx](t - Dmat)*X[to](t))
        """
        pass

    def _additive_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform additive coupling on the network within given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx] * X[idx](t - Dmat))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity: connectivity matrix - if None will use the network
            connectivity from init
        :type connectivity: np.ndarray
        """
        pass

    def _couple(self, coupling_type, coupling_variable):
        """
        Perform simple coupling in the network based on the type.

        :param coupling_type: type of the coupling - additive, diffusive or none
        :type coupling_type: str
        :param coupling_variable: perform coupling on this coupling coupling
            variable
        :type coupling_variable: str
        """
        pass

    def _derivatives(self):
        """
        Gather all derivatives from all nodes.
        """
        pass