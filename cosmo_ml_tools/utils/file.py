import os
from cobaya.yaml import yaml_load

def _find_file(filename):
    """Find the file path, first checking if it exists and then looking in the ``external`` directory."""
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'external', filename)

    if not os.path.exists(path):
        raise ValueError('Cannot locate file {}'.format(filename))

    return path

def load_ini(filename):
    """
    Read a CLASS ``.ini`` file, returning a dictionary of parameters.

    Parameters
    ----------
    filename : str
        The name of an existing parameter file to load, or one included as part of the CLASS source.

    Returns
    -------
    ini : dict
        The input parameters loaded from file.
    """
    # also look in data dir
    filename = _find_file(filename)

    params = {}
    with open(filename, 'r') as file:

        # loop over lines
        for j, line in enumerate(file):
            if not line: continue

            # skip any commented lines with #
            if '#' in line: line = line[line.index('#') + 1:]

            # must have an equals sign to be valid
            if "=" not in line: continue

            # extract key and value pairs
            fields = line.split('=')
            if len(fields) != 2:
                import warnings
                warnings.warn('Skipping line number {}: "{}"'.format(j, line))
                continue
            params[fields[0].strip()] = fields[1].strip()

    return params


def load_precision(filename):
    """
    Load a CLASS ``.pre`` file, and return a dictionary of input parameters.

    Parameters
    ----------
    filename : str
        .ini file to load passed to class.

    Returns
    -------
    pre : dict
        The precision parameters loaded from the file.
    """
    return load_ini(filename)

def load_bf(filename):
    raise NotImplementedError

def _is_yaml(filename):
    return True if '.yaml' in filename else False

def _is_ini(filename):
    return True if '.ini' in filename else False

def _is_bestfit(filename):
    return True if '.bestfit' in filename else False

def initialize_helper(ini_file:str|dict) -> dict:
    """Helper function commonly used in the __init__ method.

    Args:
        ini_file (str|dict): _description_

    Returns:
        dict: a dictionary with the information contained in the ini_file.
    """
    if isinstance(ini_file, str):
        if _is_yaml(ini_file):
            info=yaml_load(ini_file)
        elif _is_ini(ini_file):
            info=load_ini(ini_file)
        elif _is_bestfit(ini_file):
            info=load_bf(ini_file)
    elif isinstance(ini_file, dict):
        pass
    else:
        print("Unsupported type")
    return info