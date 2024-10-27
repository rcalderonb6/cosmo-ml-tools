"""
TODO: Make everything jax-dependent/compatible
"""

import numpy as np
from .base import BoltzmannBase
from .constants import *
from utils.file import _initialize_helper
from cobaya.yaml import yaml_load
# try:
#     from classy import Class
# except ModuleNotFoundError:
#     print('Class is not installed in the current environment!')

class Classy(BoltzmannBase):
    """Base Class for the Boltzmann solver Class and its extensions"""
    def __init__(self,info,other_info=None,verbose:int=0) -> None:
        
        # Handle the info variable according to the type and return a dictionary
        info=self._initialize_helper(info)
         
        self.cosmo=get_classy(info,other_info=other_info)
        self._H_units = {'1/Mpc' : 1, 
                         'km/s/Mpc' : C_KMS,
                         'dimensionless': 1 / self.cosmo.Hubble(0)}
        
        self._clean_state=True
            
    def Hubble(self,z:np.ndarray,units:str='km/s/Mpc'):
        H=np.array([self.cosmo.Hubble(zi) for zi in z])
        return self._H_units[units] * H
    
    @property
    def Cls(self,ell_factor=True,lensed=True,units='muK2'):
        return get_Cl(self.cosmo,ell_factor=ell_factor,lensed=lensed,units=units)

    @property
    def H0(self):
        return 1e2*self.cosmo.h()
    
    @property
    def Omega_g(self):
        return self.cosmo.Omega_g()
    
    @property
    def Omega_m(self):
        return self.cosmo.Omega_m()
    
    @property
    def Omega_b(self):
        return self.cosmo.Omega_b()
    
    @property
    def Omega_c(self):
        return self.cosmo.Omega0_cdm()
    
    @property
    def Omega_k(self):
        return self.cosmo.Omega0_k()
    
    
# def _classy_verbose(info,verbose=0):
#     message=[]
#     if verbose>0:
#         message+= 'Creating an instance of Class.\n'
#     if verbose>1:
#         message+= 'Requested settings are:\n'
#         message+= 'Computing observables...'
    
#     print(message)

def get_classy(info:dict,other_info:dict|None=None,verbose=0):
    """Get an instance of the Class class and compute observables requested in the `info` dictionary.

    Args:
        info (dict): common settings in a dictionary format
        other_info (dict | None, optional): Additional run-specific/precision settings. Defaults to None.

    Returns:
        Class instance: a Classy object with computations stored in it.
    """
    # _classy_verbose(verbose)
    # m=Class()
    # m.set(info)
    # if other_info is not None: m.set(other_info)
    # m.compute()
    # return m
    pass

def get_Cl(cosmo,ell_factor:bool=True,lensed:bool=True,units:str='muK2') -> dict:
    """Get the (un)lensed total CMB power spectra and lensing power spectrum

    Args:
        cosmo (_type_): An instance of the Class class
        ell_factor (bool, optional): whether to include the l(l+1)^n factor. Defaults to True.
        lensed (bool, optional): whether to retrieve the lensed- or unlensed-Cl's. Defaults to True.
        units (str, optional): FIRAS normalization to µK^2. Defaults to 'muK2'.

    Returns:
        dict: _description_
    """
    # Set the normalization to Firas T_0 measurements
    norm = T0_FIRAS**2 if units=='muK2' else 1.
    
    # Retrieve Cl's from Class
    Cls=cosmo.raw_cl() if lensed else cosmo.raw_cl()
    
    # Remove first two multipoles and scale accordingly
    l=Cls['ell'][2:]
    l_factor = {key: l*(l+1) / TWO_PI for key in ['tt','te','et','ee']}
    l_factor['pp']= (l*(l+1))**2 / TWO_PI
    l_factor.update({key: (l*(l+1))**(3/2) / TWO_PI for key in ['tp','pt','ep','pe']})
    
    return  {key: norm * l_factor[key] * val[2:] for key,val in Cls.items()}


