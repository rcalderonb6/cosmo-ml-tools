import numpy as np
from .base import BoltzmannBase
from .constants import T0_FIRAS,TWO_PI
# from classy import Class
class Classy(BoltzmannBase):
    """Base Class for the Boltzmann solver Class and its variations"""
    def __init__(self) -> None:
        pass

# def retrieve_classy(info:dict):
#     m=Class()
#     m.set(info)
#     m.compute()
#     return m

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