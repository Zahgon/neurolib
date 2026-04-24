import numpy as np
from .timeIntegration import simulateBOLD

class BOLDModel:
    """
    Balloon-Windkessel BOLD simulator class.
    BOLD activity is downsampled to 0.5 Hz by default.

    BOLD simulation results are saved in t_BOLD, BOLD instance attributes.
    """

    def __init__(self, N, dt, normalize_input=False, normalize_max=50):
        self.N = N
        self.dt = dt
        self.samplingRate_NDt = int(round(2000 / dt))
        self.normalize_input = normalize_input
        self.normalize_max = normalize_max
        self.t_BOLD = np.array([], dtype='f', ndmin=2)
        self.BOLD = np.array([], dtype='f', ndmin=2)
        self.all_Rates = np.array([], dtype='f', ndmin=2)
        self.BOLD_chunk = np.array([], dtype='f', ndmin=2)
        self.idxLastT = 0
        self.X_BOLD = np.ones((N,))
        self.F_BOLD = np.ones((N,))
        self.Q_BOLD = np.ones((N,))
        self.V_BOLD = np.ones((N,))

    def run(self, activity):
        """Runs the Balloon-Windkessel BOLD simulation.

        Parameters:
            :param activity:     Neuronal firing rate in Hz

        :param activity: Neuronal firing rate in Hz
        :type activity: numpy.ndarray
        """
        pass