import numpy as np
from jitcdde import input as system_input
from symengine import exp
from ....utils.stimulus import OrnsteinUhlenbeckProcess, ZeroInput
from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass
TCR_DEFAULT_PARAMS = {'tau': 20.0, 'Q_max': 0.4, 'theta': -58.5, 'sigma': 6.0, 'C1': 1.8137993642, 'C_m': 1.0, 'gamma_e': 0.07, 'gamma_r': 0.1, 'g_L': 1.0, 'g_GABA': 1.0, 'g_AMPA': 1.0, 'g_LK': 0.018, 'g_T': 3.0, 'g_h': 0.062, 'E_AMPA': 0.0, 'E_GABA': -70.0, 'E_L': -70.0, 'E_K': -100.0, 'E_Ca': 120.0, 'E_h': -40.0, 'alpha_Ca': -5.18e-05, 'tau_Ca': 10.0, 'Ca_0': 0.00024, 'k1': 25000000.0, 'k2': 0.0004, 'k3': 0.1, 'k4': 0.001, 'n_P': 4.0, 'g_inc': 2.0, 'ext_current': 0.0, 'lambda': LAMBDA_SPEED}
TRN_DEFAULT_PARAMS = {'tau': 20.0, 'Q_max': 0.4, 'theta': -58.5, 'sigma': 6.0, 'C1': 1.8137993642, 'C_m': 1.0, 'gamma_e': 0.07, 'gamma_r': 0.1, 'g_L': 1.0, 'g_GABA': 1.0, 'g_AMPA': 1.0, 'g_LK': 0.018, 'g_T': 2.3, 'E_AMPA': 0.0, 'E_GABA': -70.0, 'E_L': -70.0, 'E_K': -100.0, 'E_Ca': 120.0, 'ext_current': 0.0, 'lambda': LAMBDA_SPEED}
THALAMUS_NODE_DEFAULT_CONNECTIVITY = np.array([[0.0, 5.0], [3.0, 25.0]])

class ThalamicMass(NeuralMass):
    """
    Base for thalamic neural populations

    Reference:
        Costa, M. S., Weigenand, A., Ngo, H. V. V., Marshall, L., Born, J.,
        Martinetz, T., & Claussen, J. C. (2016). A thalamocortical neural mass
        model of the EEG during NREM sleep and its response to auditory stimulation.
        PLoS computational biology, 12(9).
    """
    name = 'Thalamic mass'
    label = 'THLM'

    def _get_firing_rate(self, voltage):
        pass

    def _get_excitatory_current(self, voltage, synaptic_rate):
        pass

    def _get_inhibitory_current(self, voltage, synaptic_rate):
        pass

    def _get_leak_current(self, voltage):
        pass

    def _get_potassium_leak_current(self, voltage):
        pass

    def _get_T_type_current(self, voltage, h_T_value):
        pass

class ThalamocorticalMass(ThalamicMass):
    """
    Excitatory mass representing thalamocortical relay neurons in the thalamus.
    """
    name = 'Thalamocortical relay mass'
    label = 'TCR'
    mass_type = EXC
    num_state_variables = 10
    num_noise_variables = 1
    coupling_variables = {9: f'r_mean_{EXC}'}
    required_couplings = ['node_exc_exc', 'node_exc_inh', 'network_exc_exc']
    state_variable_names = ['V', 'Ca', 'h_T', 'm_h1', 'm_h2', 's_e', 's_i', 'ds_e', 'ds_i', 'r_mean']
    required_params = ['tau', 'Q_max', 'theta', 'sigma', 'C1', 'C_m', 'gamma_e', 'gamma_r', 'g_L', 'g_GABA', 'g_AMPA', 'g_LK', 'g_T', 'g_h', 'E_AMPA', 'E_GABA', 'E_L', 'E_K', 'E_Ca', 'E_h', 'alpha_Ca', 'tau_Ca', 'Ca_0', 'k1', 'k2', 'k3', 'k4', 'n_P', 'g_inc', 'ext_current', 'lambda']
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0)]

    def __init__(self, params=None):
        super().__init__(params=params or TCR_DEFAULT_PARAMS)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        pass

    def _get_anomalous_rectifier_current(self, voltage, m_h1_value, m_h2_value):
        pass

    def _m_inf_T(self, voltage):
        pass

    def _h_inf_T(self, voltage):
        pass

    def _tau_h_T(self, voltage):
        pass

    def _m_inf_h(self, voltage):
        pass

    def _tau_m_h(self, voltage):
        pass

    def _P_h(self, ca_conc):
        pass

    def _derivatives(self, coupling_variables):
        pass

class ThalamicReticularMass(ThalamicMass):
    """
    Inhibitory mass representing thalamic reticular nuclei neurons in the
    thalamus.
    """
    name = 'Thalamic reticular nuclei mass'
    label = 'TRN'
    mass_type = INH
    num_state_variables = 7
    num_noise_variables = 1
    coupling_variables = {6: f'r_mean_{INH}'}
    required_couplings = ['node_inh_exc', 'node_inh_inh', 'network_inh_exc']
    state_variable_names = ['V', 'h_T', 's_e', 's_i', 'ds_e', 'ds_i', 'r_mean']
    required_params = ['tau', 'Q_max', 'theta', 'sigma', 'C1', 'C_m', 'gamma_e', 'gamma_r', 'g_L', 'g_GABA', 'g_AMPA', 'g_LK', 'g_T', 'E_AMPA', 'E_GABA', 'E_L', 'E_K', 'E_Ca', 'ext_current', 'lambda']
    _noise_input = [ZeroInput()]

    def __init__(self, params=None):
        super().__init__(params=params or TRN_DEFAULT_PARAMS)

    def _m_inf_T(self, voltage):
        pass

    def _h_inf_T(self, voltage):
        pass

    def _tau_h_T(self, voltage):
        pass

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        pass

    def _derivatives(self, coupling_variables):
        pass

class ThalamicNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Thalamic mass model network node with 1 excitatory (TCR) and 1 inhibitory
    (TRN) population due to Costa et al.
    """
    name = 'Thalamic mass model node'
    label = 'THLMnode'
    default_network_coupling = {'network_exc_exc': 0.0, 'network_inh_exc': 0.0}
    default_output = f'r_mean_{EXC}'
    output_vars = [f'r_mean_{EXC}', f'r_mean_{INH}', f'V_{EXC}', f'V_{INH}']

    def __init__(self, tcr_params=None, trn_params=None, connectivity=THALAMUS_NODE_DEFAULT_CONNECTIVITY):
        """
        :param tcr_params: parameters for the excitatory (TCR) mass
        :type tcr_params: dict|None
        :param trn_params: parameters for the inhibitory (TRN) mass
        :type trn_params: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        """
        tcr_mass = ThalamocorticalMass(params=tcr_params)
        tcr_mass.index = 0
        trn_mass = ThalamicReticularMass(params=trn_params)
        trn_mass.index = 1
        super().__init__(neural_masses=[tcr_mass, trn_mass], local_connectivity=connectivity, local_delays=None)