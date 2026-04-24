"""
Base classes for representing signals.
"""
import logging
from copy import deepcopy
from functools import partial
import numpy as np
import xarray as xr
from ..models.model import Model
from scipy.signal import butter, detrend, get_window, hilbert
from scipy.signal import resample as scipy_resample
from scipy.signal import sosfiltfilt
NC_EXT = '.nc'

def scipy_iir_filter_data(x, sfreq, l_freq, h_freq, l_trans_bandwidth=None, h_trans_bandwidth=None, **kwargs):
    """
    Custom, scipy based filtering function with basic butterworth filter.

    :param x: data to be filtered, time is the last axis
    :type x: np.ndarray
    :param sfreq: sampling frequency of the data in Hz
    :type sfreq: float
    :param l_freq: frequency below which to filter the data in Hz
    :type l_freq: float|None
    :param h_freq: frequency above which to filter the data in Hz
    :type h_freq: float|None
    :param l_trans_bandwidth: keeping for compatibility with mne
    :type l_trans_bandwidth: None
    :param h_trans_bandwidth: keeping for compatibility with mne
    :type h_trans_bandwidth: None
    :return: filtered data
    :rtype: np.ndarray
    """
    pass

class Signal:
    name = ''
    label = ''
    signal_type = ''
    unit = ''
    description = ''
    _copy_attributes = ['name', 'label', 'signal_type', 'unit', 'description', 'process_steps']
    PROCESS_STEPS_KEY = 'process_steps'

    @classmethod
    def from_model_output(cls, model, group='', time_in_ms=True):
        """
        Initial Signal from modelling output.
        """
        pass

    @classmethod
    def from_file(cls, filename):
        """
        Load signal from saved file.

        :param filename: filename for the Signal
        :type filename: str
        """
        pass

    def __init__(self, data, time_in_ms=False):
        """
        :param data: data for the signal, assumes time dimension with time in seconds
        :type data: xr.DataArray
        :param time_in_ms: whether time dimension is in ms
        :type time_in_ms: bool
        """
        assert isinstance(data, xr.DataArray)
        data = deepcopy(data)
        assert 'time' in data.dims, 'DataArray must have time axis'
        if time_in_ms:
            data['time'] = data['time'] / 1000.0
        data['time'] = np.around(data['time'], 6)
        self.data = data
        self.data = self.data.transpose(*self.dims_not_time + ['time'])
        self.dt = np.around(np.diff(data.time).mean(), 6)
        self.sampling_frequency = 1.0 / self.dt
        self.process_steps = [f'raw {self.signal_type} signal: {self.start_time}--{self.end_time}s']

    def __str__(self):
        """
        String representation.
        """
        return f'{self.name} representing {self.signal_type} signal with unit of {self.unit} with user-provided description: `{self.description}`. Shape of the signal is {self.shape} with dimensions {self.data.dims}. Signal starts at {self.start_time} and ends at {self.end_time}.'

    def __repr__(self):
        """
        Representation.
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Comparison operator.

        :param other: other `Signal` to compare with
        :type other: `Signal`
        :return: whether two `Signals` are the same
        :rtype: bool
        """
        assert isinstance(other, Signal)
        try:
            xr.testing.assert_allclose(self.data, other.data)
            eq = True
        except AssertionError:
            eq = False
        for attr in self._copy_attributes:
            if getattr(self, attr) != getattr(other, attr):
                logging.warning(f'`{attr}` not equal between signals.')
        return eq

    def __getitem__(self, pos):
        """
        Get item selects in output dimension.
        """
        add_steps = [f'select `{pos}` output']
        return self.__constructor__(self.data.sel(output=pos)).__finalize__(self, add_steps)

    def __finalize__(self, other, add_steps=None):
        """
        Copy attributes from other to self. Used when constructing class
        instance with different data, but same metadata.

        :param other: other instance of `Signal`
        :type other: `Signal`
        :param add_steps: add steps to preprocessing
        :type add_steps: list|None
        """
        assert isinstance(other, Signal)
        for attr in self._copy_attributes:
            setattr(self, attr, deepcopy(getattr(other, attr)))
        if add_steps is not None:
            self.process_steps += add_steps
        return self

    @property
    def __constructor__(self):
        """
        Return constructor, so that each child class would initiate a new
        instance of the correct class, i.e. first in the method resolution
        order.
        """
        return self.__class__.mro()[0]

    def _write_attrs_to_xr(self):
        """
        Copy attributes to xarray before saving.
        """
        pass

    def save(self, filename):
        """
        Save signal.

        :param filename: filename to save, currently saves to netCDF file, which is natively supported by xarray
        :type filename: str
        """
        pass

    def iterate(self, return_as='signal'):
        """
        Return iterator over columns, so univariate measures can be computed
        per column. Loops over tuples as (variable name, timeseries).

        :param return_as: how to return columns: `xr` as xr.DataArray, `signal` as
            instance of NeuroSignal with the same attributes as the mother signal
        :type return_as: str
        """
        pass

    def sel(self, sel_args, inplace=True):
        """
        Subselect part of signal using xarray's `sel`, i.e. selecting by actual
        physical index, hence time in seconds.

        :param sel_args: arguments you'd give to xr.sel(), i.e. slice of times
            you want to select, in seconds as a len=2 list or tuple
        :type sel_args: tuple|list
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def isel(self, isel_args, inplace=True):
        """
        Subselect part of signal using xarray's `isel`, i.e. selecting by index,
        hence integers.

        :param loc_args: arguments you'd give to xr.isel(), i.e. slice of
            indices you want to select, in seconds as a len=2 list or tuple
        :type loc_args: tuple|list
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def rolling(self, roll_over, function=np.mean, dropnans=True, inplace=True):
        """
        Return rolling reduction over signal's time dimension. The window is
        centered around the midpoint.

        :param roll_over: window to use, in seconds
        :type roll_over: float
        :param function: function to use for reduction
        :type function: callable
        :param dropnans: whether to drop NaNs - will shorten time dimension, or
            not
        :type dropnans: bool
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def sliding_window(self, length, step=1, window_function='boxcar', lengths_in_seconds=False):
        """
        Return iterator over sliding windows with windowing function applied.
        Each window has length `length` and each is translated by `step` steps.
        For no windowing function use "boxcar". If the last window would have
        the same length as other, it is omitted, i.e. last window does not have
        to end with the final timeseries point!

        :param length: length of the window, can be index or time in seconds,
            see `lengths_in_seconds`
        :type length: int|float
        :param step: how much to translate window in the temporal sense, can be
            index or time in seconds, see `lengths_in_seconds`
        :type step: int|float
        :param window_function: windowing function to use, this is passed to
            `get_window()`; see `scipy.signal.windows.get_window` documentation
        :type window_function: str|tuple|float
        :param lengths_in_seconds: if True, `length` and `step` are interpreted
            in seconds, if False they are indices
        :type lengths_in_seconds: bool
        :yield: generator with windowed Signals
        """
        pass

    @property
    def shape(self):
        """
        Return shape of the data. Time axis is the first one.
        """
        return self.data.shape

    @property
    def dims_not_time(self):
        """
        Return list of dimensions that are not time.
        """
        pass

    @property
    def coords_not_time(self):
        """
        Return dict with all coordinates except time.
        """
        pass

    @property
    def start_time(self):
        """
        Return starting time of the signal.
        """
        pass

    @property
    def end_time(self):
        """
        Return ending time of the signal.
        """
        pass

    @property
    def time(self):
        """
        Return time vector.
        """
        pass

    @property
    def preprocessing_steps(self):
        """
        Return preprocessing steps done on the data.
        """
        pass

    def pad(self, how_much, in_seconds=False, padding_type='constant', side='both', inplace=True, **kwargs):
        """
        Pad signal by `how_much` on given side of given type.

        :param how_much: how much we should pad, can be time points, or seconds,
            see `in_seconds`
        :type how_much: float|int
        :param in_seconds: whether `how_much` is in seconds, if False, it is
            number of time points
        :type in_seconds: bool
        :param padding_type: how to pad the signal, see `np.pad` documentation
        :type padding_type: str
        :param side: which side to pad - "before", "after", or "both"
        :type side: str
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        :kwargs: passed to `np.pad`
        """
        pass

    def normalize(self, std=False, inplace=True):
        """
        De-mean the timeseries. Optionally also standardise.

        :param std: normalize by std, i.e. to unit variance
        :type std: bool
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def resample(self, to_frequency, inplace=True):
        """
        Resample signal to target frequency.

        :param to_frequency: target frequency of the signal, in Hz
        :type to_frequency: float
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def hilbert_transform(self, return_as='complex', inplace=True):
        """
        Perform hilbert transform on the signal resulting in analytic signal.

        :param return_as: what to return
            `complex` will compute only analytical signal
            `amplitude` will compute amplitude, hence abs(H(x))
            `phase_wrapped` will compute phase, hence angle(H(x)), in -pi,pi
            `phase_unwrapped` will compute phase in a continuous sense, hence
                monotonic
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def detrend(self, segments=None, inplace=True):
        """
        Linearly detrend signal. If segments are given, detrending will be
        performed in each part.

        :param segments: segments for detrending, if None will detrend whole
            signal, given as indices of the time array
        :type segments: list|None
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

    def filter(self, low_freq, high_freq, l_trans_bandwidth='auto', h_trans_bandwidth='auto', inplace=True, **kwargs):
        """
        Filter data. Can be:
            low-pass (low_freq is None, high_freq is not None),
            high-pass (high_freq is None, low_freq is not None),
            band-pass (l_freq < h_freq),
            band-stop (l_freq > h_freq) filter type

        :param low_freq: frequency below which to filter the data
        :type low_freq: float|None
        :param high_freq: frequency above which to filter the data
        :type high_freq: float|None
        :param l_trans_bandwidth: transition band width for low frequency
        :type l_trans_bandwidth: float|str
        :param h_trans_bandwidth: transition band width for high frequency
        :type h_trans_bandwidth: float|str
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        :**kwargs: possible keywords to `mne.filter.create_filter`:
            `filter_length`="auto",
            `method`="fir",
            `iir_params`=None
            `phase`="zero",
            `fir_window`="hamming",
            `fir_design`="firwin"
        """
        pass

    def functional_connectivity(self, fc_function=np.corrcoef):
        """
        Compute and return functional connectivity from the data.

        :param fc_function: function which to use for FC computation, should
            take 2D array as space x time and convert it to space x space with
            desired measure
        """
        pass

    def apply(self, func, inplace=True):
        """
        Apply func for each timeseries.

        :param func: function to be applied for each 1D timeseries
        :type func: callable
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        pass

class VoltageSignal(Signal):
    name = 'Population mean membrane potential'
    label = 'V'
    signal_type = 'voltage'
    unit = 'mV'

class RatesSignal(Signal):
    name = 'Population firing rate'
    label = 'q'
    signal_type = 'rate'
    unit = 'Hz'

class BOLDSignal(Signal):
    name = 'Population blood oxygen level-dependent signal'
    label = 'BOLD'
    signal_type = 'bold'
    unit = '%'