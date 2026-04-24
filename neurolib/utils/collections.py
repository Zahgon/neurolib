"""
Collections of custom data structures and types.
"""
import logging
import random
import string
from collections.abc import MutableMapping
from dpath.util import delete, search
DEFAULT_STAR_SEPARATOR = '.'
FORWARD_REPLACE = {'*': 'STAR', '|': 'MINUS', '.': 'DOT'}
BACKWARD_REPLACE = {v: k for k, v in FORWARD_REPLACE.items()}

class dotdict(dict):
    """dot.notation access to dictionary attributes.

    Example:

    ```
    model.params['duration'] = 10 * 1000 # classic key-value dictionary
    model.params.duration = 10 * 10000 # easy access via dotdict
    ```

    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)

class star_dotdict(dotdict):
    """
    Supports star notation in dotdict. Nested dicts are now treated as glob.
    Also supports minus as a pipe ("|") for filtering out strings after |.

    Example:
        Wilson-Cowan node has in total four parameters named tau (time constants
        for both excitatory and inhibitory populations, and Ornstein-Uhlenbeck
        time constants for both populations), hence:

        > model.params["*tau"]
        # returns
        {'WCnode_0.WCmassEXC_0.tau': 2.5,
        'WCnode_0.WCmassEXC_0.noise_0.tau': 5.0,
        'WCnode_0.WCmassINH_1.tau': 3.75,
        'WCnode_0.WCmassINH_1.noise_0.tau': 5.0}

    Now let's imagine you want to make exploration over population time constants,
    but keep O-U as is, you can:

        > model.params["*tau|noise"]
        # returns
        {'WCnode_0.WCmassEXC_0.tau': 2.5, 'WCnode_0.WCmassINH_1.tau': 3.75}

    In other words, the string after "|" is filtered out from all the keys.
    This works with setting, getting, and deleting an item and also
    parameters defined with minus can be used in `Evolution` and
    `Exploration` classes.
    """

    def __getitem__(self, attr):
        if '|' in attr and '*' in attr:
            assert attr.count('|') == 1, f'Only one filter allowed: {attr}'
            search_attr, filter_out = attr.split('|')
            searched = self.__getitem__(search_attr)
            return {k: v for k, v in searched.items() if filter_out not in k}
        if '*' in attr:
            return search(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        else:
            return dict.get(self, attr)

    def __setitem__(self, attr, val):
        if '|' in attr and '*' in attr:
            assert attr.count('|') == 1, f'Only one filter allowed: {attr}'
            search_attr, filter_out = attr.split('|')
            attr = search_attr
        else:
            filter_out = ''.join((random.choice(string.ascii_lowercase) for _ in range(30)))
        if '*' in attr:
            for k, _ in search(self, attr, yielded=True, separator=DEFAULT_STAR_SEPARATOR):
                if filter_out not in k:
                    setattr(self, k, val)
        else:
            dict.__setitem__(self, attr, val)

    def __delitem__(self, attr):
        if '|' in attr and '*' in attr:
            assert attr.count('|') == 1, f'Only one filter allowed: {attr}'
            key_to_del = self.__getitem__(attr).keys()
            for key in key_to_del:
                dict.__delitem__(self, key)
        elif '*' in attr:
            delete(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        else:
            dict.__delitem__(self, attr)

def _sanitize_keys(key, replace_dict):
    pass

def sanitize_dot_dict(dct, replace_dict=FORWARD_REPLACE):
    """
    Make all keys identifiers - replace "*" string for a word.
    """
    pass

def unwrap_star_dotdict(dct, model, replaced_dict=False):
    """
    Unwrap star notation of parameters into full list of parameeters for a given
    model.
    E.g. params["*tau"] = 2.3 => params["mass1.tau"] = 2.3 and params["mass2.tau"] = 2.3
    """
    pass

def flatten_nested_dict(nested_dict, parent_key='', sep=DEFAULT_STAR_SEPARATOR):
    """
    Return flat dictionary from nested one using `sep` as separator between
    levels of keys.

    :param nested_dict: nested dictionary to flatten
    :type nested_dict: dict
    :param parent_key: parent key for the new key in flat dictionary
    :type parent_key: str
    :param sep: separator between levels of keys
    :type sep: str
    """
    pass

def flat_dict_to_nested(flat_dict, sep=DEFAULT_STAR_SEPARATOR):
    """
    Transform flat dictionary into nested one.

    :param flat_dict: flat dictionary with parameters, keys separated with `sep`
    :type flat_dict: dict
    :param sep: separator
    :type sep: str
    :return: nested dictionary
    :rtype: dict
    """
    pass