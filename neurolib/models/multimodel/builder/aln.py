import logging
import os
import numba
import numpy as np
import symengine as se
from h5py import File
from jitcdde import input as system_input
from ....utils.stimulus import OrnsteinUhlenbeckProcess
from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import Network, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass
DEFAULT_QUANTITIES_CASCADE_FILENAME = 'quantities_cascade.h5'
ALN_EXC_DEFAULT_PARAMS = {'Ke': 800.0, 'Ki': 200.0, 'c_gl': 0.4, 'Ke_gl': 250.0, 'tau_se': 2.0, 'tau_si': 5.0, 'sigmae_ext': 1.5, 'Jee_max': 2.43, 'Jei_max': -3.3, 'C': 200.0, 'gL': 10.0, 'ext_exc_current': 0.0, 'ext_exc_rate': 0.0, 'a': 15.0, 'b': 40.0, 'EA': -80.0, 'tauA': 200.0, 'lambda': LAMBDA_SPEED}
ALN_INH_DEFAULT_PARAMS = {'Ke': 800.0, 'Ki': 200.0, 'c_gl': 0.4, 'Ke_gl': 250.0, 'tau_se': 2.0, 'tau_si': 5.0, 'sigmai_ext': 1.5, 'Jie_max': 2.6, 'Jii_max': -1.64, 'C': 200.0, 'gL': 10.0, 'ext_inh_current': 0.0, 'ext_inh_rate': 0.0, 'lambda': LAMBDA_SPEED}
ALN_NODE_DEFAULT_CONNECTIVITY = np.array([[0.3, 0.5], [0.3, 0.5]])
ALN_NODE_DEFAULT_DELAYS = np.array([[4.0, 2.0], [4.0, 2.0]])

@numba.njit()
def _get_interpolation_values(xi, yi, sigma_range, mu_range, d_sigma, d_mu):
    """
    Return values needed for interpolation: bilinear (2D) interpolation
    within ranges, linear (1D) if "one edge" is crossed, corner value if
    "two edges" are crossed. Defined as jitted function due to compatibility
    with numba backend.

    :param xi: interpolation value on x-axis, i.e. I_sigma
    :type xi: float
    :param yi: interpolation value on y-axis, i.e. I_mu
    :type yi: float
    :param sigma_range: range of x-axis, i.e. sigma values
    :type sigma_range: np.ndarray
    :param mu_range: range of y-axis, i.e. mu values
    :type mu_range: np.ndarray
    :param d_sigma: grid coarsness in the x-axis, i.e. sigma values
    :type d_sigma: float
    :param d_mu: grid coarsness in the y-axis, i.e. mu values
    :type d_mu: float
    :return: index of the lower interpolation value on x-axis, index of the
        lower interpolation value on y-axis, distance of xi to the lower
        value, distance of yi to the lower value
    :rtype: (int, int, float, float)
    """
    pass

@numba.njit()
def _table_lookup(current_mu, current_sigma, sigma_range, mu_range, d_sigma, d_mu, transfer_function_table):
    """
    Translate mean and std. deviation of the current to selected quantity using
    linear-nonlinear lookup table for ALN. Defined as jitted function due to
    compatibility with numba backend.
    """
    pass

class ALNMass(NeuralMass):
    """
    Adaptive linear-nonlinear neural mass model of exponential integrate-and-fire (AdEx)
    neurons.

    References:
        Cakan C., Obermayer K. (2020). Biophysically grounded mean-field models of
        neural populations under electrical stimulation. PLoS Comput Biol, 16(4),
        e1007822.

        Augustin, M., Ladenbauer, J., Baumann, F., & Obermayer, K. (2017).
        Low-dimensional spike rate models derived from networks of adaptive
        integrate-and-fire neurons: comparison and implementation. PLoS Comput Biol,
        13(6), e1005545.
    """
    name = 'ALN neural mass model'
    label = 'ALNMass'
    python_callbacks = ['firing_rate_lookup', 'voltage_lookup', 'tau_lookup']
    num_noise_variables = 1

    def __init__(self, params, lin_nonlin_transfer_function_filename=None, seed=None):
        """
        :param lin_nonlin_transfer_function_filename: filename for precomputed
            transfer functions of the ALN model, if None, will look for it in this
            directory
        :type lin_nonlin_transfer_function_filename: str|None
        :param seed: seed for random number generator
        :type seed: int|None
        """
        super().__init__(params=params, seed=seed)
        lin_nonlin_transfer_function_filename = lin_nonlin_transfer_function_filename or os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'aln', 'aln-precalc', DEFAULT_QUANTITIES_CASCADE_FILENAME)
        self._load_lin_nonlin_transfer_function(lin_nonlin_transfer_function_filename)

    def _load_lin_nonlin_transfer_function(self, filename):
        """
        Load precomputed transfer functions from h5 file.
        """
        pass

    def describe(self):
        pass

    def _callbacks(self):
        """
        Construct list of python callbacks for ALN model.
        """
        pass

    def _numba_callbacks(self):
        """
        Define numba callbacks - has to be different than jitcdde callbacks
        because of the internals.
        """
        pass

    def firing_rate_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to firing rate using
        linear-nonlinear lookup table for ALN.
        """
        pass

    def voltage_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to voltage using
        precomputed transfer functions of the aln model.
        """
        pass

    def tau_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to tau - membrane time
        constant using precomputed transfer functions of the aln model.
        """
        pass

    def _get_current_sigma(self, I_syn_sigma_exc, I_syn_sigma_inh, exc_inp, inh_inp, J_exc_max, J_inh_max, ext_sigma):
        """
        Compute membrane current standard deviation sigma.
        """
        pass

    def _get_synaptic_current_mu(self, I_syn_mu, inp, tau):
        """
        Compute synaptic current mean mu. Used for both excitatory and inhibitory cuurent.
        """
        pass

    def _get_synaptic_current_sigma(self, I_syn_mu, I_syn_sigma, inp, inp_sq, tau):
        pass

class ExcitatoryALNMass(ALNMass):
    """
    Excitatory ALN neural mass. Contains firing rate adaptation current.
    """
    name = 'ALN excitatory neural mass'
    label = f'ALNMass{EXC}'
    num_state_variables = 7
    coupling_variables = {6: f'r_mean_{EXC}'}
    mass_type = EXC
    state_variable_names = ['I_mu', 'I_A', 'I_syn_mu_exc', 'I_syn_mu_inh', 'I_syn_sigma_exc', 'I_syn_sigma_inh', 'r_mean']
    required_couplings = ['node_exc_exc', 'node_exc_exc_sq', 'node_exc_inh', 'node_exc_inh_sq', 'network_exc_exc', 'network_exc_exc_sq']
    required_params = ['Ke', 'Ki', 'c_gl', 'Ke_gl', 'tau_se', 'tau_si', 'sigmae_ext', 'Jee_max', 'Jei_max', 'C', 'ext_exc_current', 'ext_exc_rate', 'a', 'b', 'EA', 'tauA', 'lambda']
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.4, sigma=0.0, tau=5.0)]

    def __init__(self, params=None, lin_nonlin_transfer_function_filename=None, seed=None):
        super().__init__(params=params or ALN_EXC_DEFAULT_PARAMS, lin_nonlin_transfer_function_filename=lin_nonlin_transfer_function_filename, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        pass

    def _get_adaptation_current(self, I_adaptation, firing_rate, voltage):
        """
        Compute adaptation current as a sum of subthreshold adaptation and spike-triggered adaptation.
        """
        pass

    def _compute_couplings(self, coupling_variables):
        """
        Helper that computes coupling from other nodes and network.
        """
        pass

    def _derivatives(self, coupling_variables):
        pass

class InhibitoryALNMass(ALNMass):
    """
    Inhibitory ALN neural mass. In contrast to excitatory, inhibitory mass do
    not contain fiting rate adaptation current.
    """
    name = 'ALN inhibitory neural mass'
    label = f'ALNMass{INH}'
    num_state_variables = 6
    coupling_variables = {5: f'r_mean_{INH}'}
    mass_type = INH
    state_variable_names = ['I_mu', 'I_syn_mu_exc', 'I_syn_mu_inh', 'I_syn_sigma_exc', 'I_syn_sigma_inh', 'r_mean']
    required_couplings = ['node_inh_exc', 'node_inh_exc_sq', 'node_inh_inh', 'node_inh_inh_sq']
    required_params = ['Ke', 'Ki', 'c_gl', 'Ke_gl', 'tau_se', 'tau_si', 'sigmai_ext', 'Jie_max', 'Jii_max', 'C', 'ext_inh_current', 'ext_inh_rate', 'lambda']
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.3, sigma=0.0, tau=5.0)]

    def __init__(self, params=None, lin_nonlin_transfer_function_filename=None, seed=None):
        super().__init__(params=params or ALN_INH_DEFAULT_PARAMS, lin_nonlin_transfer_function_filename=lin_nonlin_transfer_function_filename, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        pass

    def _compute_couplings(self, coupling_variables):
        """
        Helper that computes coupling from other nodes and network.
        """
        pass

    def _derivatives(self, coupling_variables):
        pass

class ALNNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default ALN network node with 1 excitatory (featuring adaptive current) and
    1 inhibitory population.
    """
    name = 'ALN neural mass node'
    label = 'ALNNode'
    sync_variables = ['node_exc_exc', 'node_inh_exc', 'node_exc_inh', 'node_inh_inh', 'node_exc_exc_sq', 'node_inh_exc_sq', 'node_exc_inh_sq', 'node_inh_inh_sq']
    default_network_coupling = {'network_exc_exc': 0.0, 'network_exc_exc_sq': 0.0}
    default_output = f'r_mean_{EXC}'
    output_vars = [f'r_mean_{EXC}', f'r_mean_{INH}', f'I_A_{EXC}']

    def _rescale_connectivity(self):
        """
        Rescale connection strengths for ALN. Should work also for ALN nodes
        with arbitrary number of masses of any type.
        """
        pass

    def __init__(self, exc_params=None, inh_params=None, exc_lin_nonlin_transfer_function_filename=None, inh_lin_nonlin_transfer_function_filename=None, connectivity=ALN_NODE_DEFAULT_CONNECTIVITY, delays=ALN_NODE_DEFAULT_DELAYS, exc_seed=None, inh_seed=None):
        """
        :param exc_params: parameters for the excitatory mass
        :type exc_params: dict|None
        :param inh_params: parameters for the inhibitory mass
        :type inh_params: dict|None
        :param exc_lin_nonlin_transfer_function_filename: filename for precomputed
            linear-nonlinear transfer functions for excitatory ALN mass, if None, will
            look for it in this directory
        :type exc_lin_nonlin_transfer_function_filename: str|None
        :param inh_lin_nonlin_transfer_function_filename: filename for precomputed
            linear-nonlinear transfer functions for inhibitory ALN mass, if None, will
            look for it in this directory
        :type inh_lin_nonlin_transfer_function_filename: str|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        :param delays: local delay matrix
        :type delays: np.ndarray
        :param exc_seed: seed for random number generator for the excitatory
            mass
        :type exc_seed: int|None
        :param inh_seed: seed for random number generator for the inhibitory
            mass
        :type inh_seed: int|None
        """
        excitatory_mass = ExcitatoryALNMass(params=exc_params, lin_nonlin_transfer_function_filename=exc_lin_nonlin_transfer_function_filename, seed=exc_seed)
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryALNMass(params=inh_params, lin_nonlin_transfer_function_filename=inh_lin_nonlin_transfer_function_filename, seed=inh_seed)
        inhibitory_mass.index = 1
        super().__init__(neural_masses=[excitatory_mass, inhibitory_mass], local_connectivity=connectivity, local_delays=delays)
        self._rescale_connectivity()

    def update_params(self, params_dict, rescale=True):
        """
        Rescale connectivity after params update if connectivity was updated.
        """
        pass

    def _sync(self):
        """
        Apart from basic EXC<->INH connectivity, construct also squared
        variants.
        """
        pass

class ALNNetwork(Network):
    """
    Whole brain network of adaptive exponential integrate-and-fire mean-field
    excitatory and inhibitory nodes.
    """
    name = 'ALN neural mass network'
    label = 'ALNNet'
    sync_variables = ['network_exc_exc', 'network_exc_exc_sq']
    output_vars = [f'r_mean_{EXC}', f'r_mean_{INH}', f'I_A_{EXC}']

    def __init__(self, connectivity_matrix, delay_matrix, exc_mass_params=None, inh_mass_params=None, exc_lin_nonlin_transfer_function_filename=None, inh_lin_nonlin_transfer_function_filename=None, local_connectivity=ALN_NODE_DEFAULT_CONNECTIVITY, local_delays=ALN_NODE_DEFAULT_DELAYS, exc_seed=None, inh_seed=None):
        """
        :param connectivity_matrix: connectivity matrix for coupling between
             nodes, defined as [from, to]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, if None, delays are
        all zeros, in ms, defined as [from, to]
        :type delay_matrix: np.ndarray|None
        :param exc_mass_params: parameters for each excitatory ALN neural
            mass, if None, will use default
        :type exc_mass_params: list[dict]|dict|None
        :param inh_mass_params: parameters for each inhibitory ALN neural
            mass, if None, will use default
        :type inh_mass_params: list[dict]|dict|None
        param exc_lin_nonlin_transfer_function_filename: filename for precomputed
            linear-nonlinear transfer_function for excitatory ALN mass, if None, will
            look for it in this directory
        :type exc_lin_nonlin_transfer_function_filename: list[str]|str|None
        :param inh_lin_nonlin_transfer_function_filename: filename for precomputed
            linear-nonlinear transfer_function for inhibitory ALN mass, if None, will
            look for it in this directory
        :type inh_lin_nonlin_transfer_function_filename: list[str]|str|None
        :param local_connectivity: local within-node connectivity matrix
        :type local_connectivity: np.ndarray
        :param local_delays: local within-node delay matrix
        :type local_delays: list[np.ndarray]|np.ndarray
        :param exc_seed: seed for random number generator for the excitatory
            masses
        :type exc_seed: int|None
        :param inh_seed: seed for random number generator for the excitatory
            masses
        :type inh_seed: int|None
        """
        num_nodes = connectivity_matrix.shape[0]
        exc_mass_params = self._prepare_mass_params(exc_mass_params, num_nodes)
        inh_mass_params = self._prepare_mass_params(inh_mass_params, num_nodes)
        exc_lin_nonlin_transfer_function_filename = self._prepare_mass_params(exc_lin_nonlin_transfer_function_filename, num_nodes, native_type=str)
        inh_lin_nonlin_transfer_function_filename = self._prepare_mass_params(inh_lin_nonlin_transfer_function_filename, num_nodes, native_type=str)
        local_connectivity = self._prepare_mass_params(local_connectivity, num_nodes, native_type=np.ndarray)
        local_delays = self._prepare_mass_params(local_delays, num_nodes, native_type=np.ndarray)
        exc_seeds = self._prepare_mass_params(exc_seed, num_nodes, native_type=int)
        inh_seeds = self._prepare_mass_params(inh_seed, num_nodes, native_type=int)
        nodes = []
        for i, (exc_params, inh_params, exc_transfer_function, inh_transfer_function, local_conn, local_dels) in enumerate(zip(exc_mass_params, inh_mass_params, exc_lin_nonlin_transfer_function_filename, inh_lin_nonlin_transfer_function_filename, local_connectivity, local_delays)):
            node = ALNNode(exc_params=exc_params, inh_params=inh_params, exc_lin_nonlin_transfer_function_filename=exc_transfer_function, inh_lin_nonlin_transfer_function_filename=inh_transfer_function, connectivity=local_conn, delays=local_dels, exc_seed=exc_seeds[i], inh_seed=inh_seeds[i])
            node.index = i
            node.idx_state_var = i * node.num_state_variables
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
            nodes.append(node)
        super().__init__(nodes=nodes, connectivity_matrix=connectivity_matrix, delay_matrix=delay_matrix)
        assert len(self.sync_variables) == 2

    def _sync(self):
        """
        Overload sync method since the ALN model requires
        squared coupling weights and non-trivial coupling indices.
        """
        pass