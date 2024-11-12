import sys,os
import numpy as np
from functools import partial
from cobaya.yaml import yaml_load
from typing import Optional, Union
    
class FileTypeNotSupported(Exception):
    pass
    
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

def load_yaml(filename:str,class_format=False):
    info=yaml_load()
    if class_format:
        tmp_info=info.copy()
        info=tmp_info['params']
        if 'extra_params' in tmp_info['theory']['classy'].keys():
            info.update(tmp_info['theory']['classy']['extra_params'])
    return info
    
def load_bf(filename:str):
    raise NotImplementedError

def load_param(filename:str):
    raise NotImplementedError

def write_bf(bestfit_point:np.ndarray,
             param_names: Optional[list[str]] = None,
             out_filename: str = 'chain.bestfit', 
             override: bool = False):
    """
    Write a GetDist-like .bestfit file

    Args:
        bestfit_point (np.ndarray): array with bestfit values for the parameters.
        param_names (list[str]|None, optional): a list with the parameter names. Defaults to None, in which case the labels param1,param2,etc are given.
        out_filename (str, optional): _description_. Defaults to 'chain.bestfit'.
        override (bool,optional): check for existing .bestfit file and override if True. Defaults to False.
    """
    if param_names is None: param_names=[f'param{i}' for i in range(len(bestfit_point))]
    if override:
        np.savetxt(out_filename,bestfit_point,header=param_names)
    else:
        print('Found existing .bestfile file. Choose another output location/filename or set `override=True` if you want to override the previously stored .bestfit file')

def _is_yaml(filename):
    return True if '.yaml' in filename else False

def _is_ini(filename):
    return True if '.ini' in filename else False

def _is_bestfit(filename):
    return True if '.bestfit' in filename else False

def _is_param(filename):
    return True if '.param' in filename else False

def _get_ini_file_type(ini_file:str):
    """
    Helper function to retrieve the initialization file type

    Args:
        ini_file (str): a string containing the initialization filename/location on disk. 
        Formats currently supported are: .param, .yaml, .ini, and .bestfit

    Returns:
        str: a string determining the type of initialization file
    """
    # .yaml input handling
    if _is_yaml(ini_file):
        fmt='yaml'
    # .ini input handling
    elif _is_ini(ini_file):
        fmt='ini'
    # .param input handling    
    elif _is_param(ini_file):
        fmt='mp'
    # .bestfit input handling    
    elif _is_bestfit(ini_file):
        fmt='bf' 
    else:
        print('Initialization file format not found')
    return fmt

def initialize_helper(ini_file:Union[str,dict],engine='class') -> dict:
    """Helper function commonly used in the various __init__ methods.

    Args:
        ini_file (str|dict): The path to an initialization file. Supports .yaml file from Cobaya and .param files from Montepython, as well as .bestfit files from Getdist.
        engine (str): The engine to assist. Defaults to 'class'.

    Returns:
        dict: a dictionary with the relevant information contained in the ini_file.
    """
    _classy = True if engine in ['classy','class'] else False
    _file_loader={'yaml': partial(load_yaml,class_format=_classy),'ini':load_ini,'bf':load_bf,'mp':load_param}
    
    # If its a string, load the corresponding file from the disk according to its format
    if isinstance(ini_file, str):
        fmt=_get_ini_file_type(ini_file)
        return _file_loader[fmt](ini_file)
    
    # If its a dictionary, its already in the correct format and we do nothing
    elif not isinstance(ini_file, dict):
        raise FileTypeNotSupported 
    
    return ini_file