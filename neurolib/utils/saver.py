"""
Saving model output.
"""
import json
import pickle
from copy import deepcopy
import os
import numpy as np
import xarray as xr

def save_to_pickle(datafield, filename):
    """
    Save datafield to pickle file. Keep in mind that restoring a pickle
    requires that the internal structure of the types for the pickled data
    remain unchanged, o.e. not recommended for long-term storage.

    :param datafield: datafield or dataarray to save
    :type datafield: xr.Dataset|xr.DataArray
    :param filename: filename
    :type filename: str
    """
    pass

def save_to_netcdf(datafield, filename):
    """
    Save datafield to NetCDF. NetCDF cannot handle structured attributes,
    hence they are stripped and if there are some, they are saved as json
    with the same filename.

    :param datafield: datafield or dataarray to save
    :type datafield: xr.Dataset|xr.DataArray
    :param filename: filename
    :type filename: str
    """
    pass

def _save_attrs_json(attrs, filename):
    """
    Save attributes to json.

    :param attrs: attributes to save
    :type attrs: dict
    :param filename: filename for the json file
    :type filename: str
    """
    pass