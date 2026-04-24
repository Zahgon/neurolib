import numba
import numpy as np
from neurolib.control.optimal_control.oc import OC
from neurolib.models.aln.timeIntegration import compute_hx, compute_hx_nw, Duh, Dxdoth, compute_hx_de, compute_hx_di

class OcAln(OC):
    """Class for optimal control specific to neurolib's implementation of the two-population ALN model
            ("ALNmodel").

    :param model: Instance of ALN model (can describe a single ALN node or a network of coupled
                  ALN nodes.
    :type model: neurolib.models.aln.model.ALNModel
    """

    def __init__(self, model, target, weights=None, print_array=[], cost_interval=(None, None), control_interval=(None, None), cost_matrix=None, control_matrix=None, M=1, M_validation=0, validate_per_step=False):
        super().__init__(model, target, weights=weights, print_array=print_array, cost_interval=cost_interval, cost_matrix=cost_matrix, control_interval=control_interval, control_matrix=control_matrix, M=M, M_validation=M_validation, validate_per_step=validate_per_step)
        assert self.model.name == 'aln'
        self.fullstate = self.get_fullstate()
        if self.model.params.filter_sigma:
            print('NOT IMPLEMENTED FOR FILTER_SIGMA=TRUE')
            raise NotImplementedError
        self.ndt_de = np.around(self.model.params.de / self.dt).astype(int)
        self.ndt_di = np.around(self.model.params.di / self.dt).astype(int)
        self.precomp_factors = self.get_precomp_factors()

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables.

        :return:        N x V x V array
        :rtuype:        np.ndarray"""
        pass

    def get_model_params(self):
        """Model params as an ordered tuple.

        :return:    22 ordered parameters
        :rtype:     tuple
        """
        pass

    def get_precomp_factors(self):
        """Precomputed factors as an ordered tuple.

        :return:    12 prefactors
        :rtype:     tuple
        """
        pass

    def Duh(self):
        """Jacobian of systems dynamics wrt. external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """
        pass

    def compute_hx_list(self):
        """List of Jacobians without and with time delays (e.g. in the ALN model) and list of respective time step delays as integers (0 for undelayed)

        :return:        List of Jacobian matrices, list of time step delays
        : rtype:        List of np.ndarray, List of integers

        """
        pass

    def compute_hx(self):
        """Jacobians of ALNModel wrt. the 'e'- and 'i'-variable for each time step.

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """
        pass

    def compute_hx_de(self):
        """Jacobians of ALNModel wrt. the variables delayed by de

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """
        pass

    def compute_hx_di(self):
        """Jacobians of ALNModel wrt. the variables delayed by di

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """
        pass

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x V x V array
        :rtype: np.ndarray
        """
        pass

    def get_fullstate(self):
        """Compute the full state (all 16 variables) of the ALN model by stepwise forward integration.

        :return:    N x V x T state vector
        :rtype:     np.ndarray
        """
        pass

    def setasinit(self, fullstate, t):
        """Set the initial state of the ALN model as defined by input 'fullstate'

        :param fullstate:   state vector to read initial state from
        :type fullstate:    np.ndarray
        :param t:           time index
        :type t:            int
        """
        pass

    def getinitstate(self):
        """Read the initial state of the ALN model

        :return:            initial state of the ALN model as N x V x N_maxdelay array
        :rtype t:           np.ndarray
        """
        pass

    def getfinalstate(self):
        """Read the final state of the ALN model (only last timestep)

        :return:            final state of the ALN model as N x V matrix
        :rtype t:           np.ndarray
        """
        pass

    def setinitstate(self, state):
        """Set the initial state of the ALN model as defined by final values of state

        :param state:       state vector to read initial state from
        :type state:        np.ndarray
        """
        pass