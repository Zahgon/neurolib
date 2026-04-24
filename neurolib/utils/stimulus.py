"""
Functions for creating stimuli and noise inputs for models.
"""
import inspect
import logging
import numba
import numpy as np
from chspy import CubicHermiteSpline
from ..models.model import Model
from scipy.signal import square

class Input:
    """
    Generates input to model.

    Base class for other input types.
    """

    def __init__(self, n=1, seed=None):
        """
        :param n: Number of spatial dimensions / independent realizations of the input.
            For determinstic inputs, the array is just copied,
            for stociastic / noisy inputs, this means independent realizations.
        :type n: int
        :param seed: Seed for the random number generator.
        :type seed: int|None
        """
        self.n = n
        self.seed = seed
        np.random.seed(seed)
        self.param_names = inspect.getfullargspec(self.__init__).args
        self.param_names.remove('self')

    def __add__(self, other):
        """
        Sum two inputs into one SummedStimulus.
        """
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, SummedStimulus):
            return SummedStimulus(inputs=[self] + other.inputs)
        else:
            return SummedStimulus(inputs=[self, other])

    def __and__(self, other):
        """
        Concatenate two inputs into ConcatenatedStimulus.
        """
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, ConcatenatedStimulus):
            return ConcatenatedStimulus(inputs=[self] + other.inputs, length_ratios=[1] + other.length_ratios)
        else:
            return ConcatenatedStimulus(inputs=[self, other])

    def _reset(self):
        """
        Reset is called after generating an input. Can be used to reset
        intrinsic properties.
        """
        pass

    def get_params(self):
        """
        Return the parameters of the input as dict.
        """
        pass

    def update_params(self, params_dict):
        """
        Update model input parameters.

        :param params_dict: New parameters for this input
        :type params_dict: dict
        """
        pass

    def _get_times(self, duration, dt):
        """
        Generate time vector.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        pass

    def generate_input(self, duration, dt):
        """
        Function to generate input.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        pass

    def as_array(self, duration, dt):
        """
        Return input as numpy array.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        pass

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        """
        Return as cubic Hermite splines.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        :param shift_start_time: By how much to shift the stimulus start time
        :type shift_start_time: float
        """
        pass

    def to_model(self, model):
        """
        Return numpy array of stimuli based on model parameters.

        Example:
        ```
        model.params["ext_exc_input"] = SinusoidalInput(...).to_model(model)
        ```

        :param model: neurolib's model
        :type model: `neurolib.models.Model`
        """
        pass

class Stimulus(Input):
    """
    Generates a stimulus with optional start and end times.
    """

    def __init__(self, start=None, end=None, n=1, seed=None):
        """
        :param start: start of the stimulus, in milliseconds
        :type start: float
        :param end: end of the stimulus, in milliseconds
        :type end: float
        """
        self.start = start
        self.end = end
        self._default_start = start
        self._default_end = end
        super().__init__(n=n, seed=seed)

    def _reset(self):
        pass

    def _get_times(self, duration, dt):
        pass

    def _trim_stim(self, stim_input):
        """
        Trim stimulus. Translate the start of the stimulus by
        padding the beginning and replace the end with zeros.
        """
        pass

class BaseMultipleInputs(Stimulus):
    """
    Base class for stimuli consisting of multiple time series, such as summed inputs or concatenated inputs.
    """

    def __init__(self, inputs):
        """
        :param inputs: List of Inputs to combine
        :type inputs: list[`Input`]
        """
        assert all((isinstance(input, Input) for input in inputs))
        self.inputs = inputs

    def __len__(self):
        """
        Return number of inputs.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Return inputs by index. This also allows iteration.
        """
        return self.inputs[index]

    @property
    def n(self):
        pass

    @n.setter
    def n(self, n):
        pass

    def get_params(self):
        """
        Get all parameters recursively for all inputs.
        """
        pass

    def update_params(self, params_dict):
        """
        Update all parameters recursively.
        """
        pass

class SummedStimulus(BaseMultipleInputs):
    """
    Represents the summation of arbitrary many stimuli.

    Example:
    ```
        summed_stimulus = SinusoidalInput(...) + OrnsteinUhlenbeckProcess(...)
    ```
    """

    def __add__(self, other):
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, SummedStimulus):
            return SummedStimulus(inputs=self.inputs + other.inputs)
        else:
            return SummedStimulus(inputs=self.inputs + [other])

    def as_array(self, duration, dt):
        """
        Return sum of all inputes as numpy array.
        """
        pass

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        """
        Return sum of all inputes as cubic Hermite splines.
        """
        pass

class ConcatenatedStimulus(BaseMultipleInputs):
    """
    Represents temporal concatenation of of arbitrary many stimuli.

    Example:
    ```
        summed_stimulus = SinusoidalInput(...) & OrnsteinUhlenbeckProcess(...)
    ```
    """

    def __init__(self, inputs, length_ratios=None):
        """
        :param length_ratios: Ratios of lengths of concatenated stimuli
        :type length_ratios: list[int|float]
        """
        if length_ratios is None:
            length_ratios = [1] * len(inputs)
        assert len(inputs) == len(length_ratios)
        assert all((length > 0 for length in length_ratios))
        self.length_ratios = length_ratios
        super().__init__(inputs)

    def __and__(self, other):
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, ConcatenatedStimulus):
            return ConcatenatedStimulus(inputs=self.inputs + other.inputs, length_ratios=self.length_ratios + other.length_ratios)
        else:
            return ConcatenatedStimulus(inputs=self.inputs + [other], length_ratios=self.length_ratios + [1])

    def as_array(self, duration, dt):
        """
        Return concatenation of all stimuli as numpy array.
        """
        pass

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        pass

class ZeroInput(Input):
    """
    No stimulus, i.e. all zeros. Can be used to add a delay between two stimuli.
    """

    def generate_input(self, duration, dt):
        pass

class WienerProcess(Input):
    """
    Stimulus sampled from a Wiener process, i.e. drawn from standard normal distribution N(0, sqrt(dt)).
    """

    def generate_input(self, duration, dt):
        pass

class OrnsteinUhlenbeckProcess(Input):
    """
    Ornstein–Uhlenbeck input, i.e.
        dX = (mu - X)/tau * dt + sigma*dW
    """

    def __init__(self, mu, sigma, tau, n=1, seed=None):
        """
        :param mu: Drift of the OU process
        :type mu: float
        :param sigma: Standard deviation of the Wiener process, i.e. strength of the noise
        :type sigma: float
        :param tau: Timescale of the OU process, in ms
        :type tau: float
        """
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        super().__init__(n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

    @staticmethod
    @numba.njit()
    def numba_ou(x, times, dt, mu, sigma, tau, n):
        """
        Generation of Ornstein-Uhlenback input - wrapped in numba's jit for
        speed.
        """
        pass

class StepInput(Stimulus):
    """
    Step input.
    """

    def __init__(self, step_size, start=None, end=None, n=1, seed=None):
        """
        :param step_size: Size of the step, i.e., the amplitude.
        :type step_size: float
        """
        self.step_size = step_size
        super().__init__(start=start, end=end, n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

class SinusoidalInput(Stimulus):
    """
    Sinusoidal input.
    """

    def __init__(self, amplitude, frequency, dc_bias=False, start=None, end=None, n=1, seed=None):
        """
        :param amplitude: Amplitude of the sinusoid.
        :type amplitude: float
        :param frequency: Frequency of the sinus oscillation, in Hz
        :type frequency: float
        :param dc_bias: Whether the sinusoid oscillates around 0
            (False), or has a positive DC bias, thus non-negative (True).
        :type dc_bias: bool
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.dc_bias = dc_bias
        super().__init__(start=start, end=end, n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

class SquareInput(Stimulus):
    """
    Oscillatory square input.
    """

    def __init__(self, amplitude, frequency, dc_bias=False, start=None, end=None, n=1, seed=None):
        """
        :param amplitude: Amplitude of the square
        :type amplitude: float
        :param frequency: Frequency of the square oscillation, in Hz
        :type frequency: float
        :param dc_bias: Whether the square oscillates around 0
            (False), or has a positive DC bias, thus non-negative (True).
        :type dc_bias: bool
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.dc_bias = dc_bias
        super().__init__(start=start, end=end, n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

class LinearRampInput(Stimulus):
    """
    Linear ramp input.
    """

    def __init__(self, inp_max, ramp_length, start=None, end=None, n=1, seed=None):
        """
        :param inp_max: Maximum of stimulus.
        :type inp_max: float
        :param ramp_length: Duration of linear ramp, in milliseconds
        :type ramp_length: float
        """
        self.inp_max = inp_max
        self.ramp_length = ramp_length
        super().__init__(start=start, end=end, n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

class ExponentialInput(Stimulus):
    """
    Exponential rise or decay input.
    """

    def __init__(self, inp_max, exp_coef=30.0, exp_type='rise', start=None, end=None, n=1, seed=None):
        """
        :param inp_max: Maximum of stimulus.
        :type inp_max: float
        :param exp_coeficient: Coeffiecent for the exponential (the higher the
            coefficient, the faster it rises or decays).
        :type exp_coeficient: float
        :param exp_type: Whether to "rise" or to "decay".
        :type exp_type: str
        """
        self.inp_max = inp_max
        self.exp_coef = exp_coef
        assert exp_type in ['rise', 'decay']
        self.exp_type = exp_type
        super().__init__(start=start, end=end, n=n, seed=seed)

    def generate_input(self, duration, dt):
        pass

def RectifiedInput(amplitude, n=1):
    """
    Return rectified input with exponential decay, i.e. a negative step followed by a
    slow decay to zero, followed by a positive step and again a slow decay to zero.
    Can be used for bistablity detection.

    :param amplitude: Amplitude (both negative and positive) for the step
    :type amplitude: float
    :param n: Number of realizations (spatial dimension)
    :type n: int
    :return: Concatenated input which represents the rectified stimulus with exponential decay
    :rtype: `ConctatenatedInput`
    """
    pass