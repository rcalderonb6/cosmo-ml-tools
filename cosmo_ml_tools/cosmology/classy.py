"""
TODO: Make everything jax-dependent/compatible
"""
import numpy as np

from .base import BoltzmannBase
from .constants import *
from ..utils.file import initialize_helper

# try:
#     from classy import Class
# except ModuleNotFoundError:
#     print('Class is not installed in the current environment!')

class ClassEngine(BoltzmannBase):
    """Base Class for the Boltzmann solver Class and its extensions"""
    
    def __init__(self,info:str|dict|None=None,cosmo=None,other_info=dict|None,verbose:int=0,name:str='name') -> None:
        """A wrapper for the Boltzmann solvers Class and its extensions.

        Args:
            info (str | dict): a string pointing to a .ini/.yaml file or python dictionary with the desired class settings/parameters
            other_info (dict|None, optional): another set of settings passed to class (e.g. precision settings). Defaults to None.
            verbose (int, optional): Print useful information for debugging purposes. Defaults to 0.
            name (str, optional): Give a name to the instance of the class.
        """
        
        self._clean_state = True
        self._name = name
        
        # Handle the info variable according to the type and return a dictionary
        self.info=initialize_helper(info)
        
        if cosmo is None:
            self.cosmo = get_classy(self.info,other_info=other_info)
        else:
            self.cosmo = cosmo
            self.update(self.info)
        
        self._H_units = {'1/Mpc' : 1, 
                         'km/s/Mpc' : C_KMS,
                         'dimensionless': 1 / self.cosmo.Hubble(0)}
        
    def Hubble(self,z:float|np.ndarray,units:str='km/s/Mpc'):
        H=np.array([self.cosmo.Hubble(zi) for zi in z]) if isinstance(z,np.ndarray) else self.cosmo.Hubble(z)
        return self._H_units[units] * H
    
    def alpha(self,which:str='M'):
        return self._alphas[which]
    
    def compute(self):
        self.cosmo.compute()
    
    def empty(self):
        self.cosmo.empty()  
        self.cosmo.cleanup_struct()
    
    def update(self,info:dict) -> None:
        """
        Update the values of the cosmological parameters with the provided dictionary and recompute observables.
        """
        self.cosmo.set(info)
        self.compute()
        self._clean_state=False
    
    def store(self):
        pass
    
    def getInfo(self):
        pass
    
    def plot(self,observables:list[str],ax=None):
        pass
    
    def _background(self):
        return self.cosmo.get_background()
    
    def _alphas(self):
        return get_alphas(self.cosmo)
    
    def Omega_of_z(self,component:str):
        _densities={'cdm':self.rho_cdm,'c':self.rho_cdm,'b':self.rho_b,'cb':self.rho_b+self.rho_cdm,'m':self.rho_m,'g':self.rho_g,'ur':self.rho_ur,'de':self.rho_de}
        Omega_of_z = _densities[component.lower()] / self.rho_crit
        return Omega_of_z
    
    @property
    def fde(self):
        return self.rho_de/self.rho_de[-1]
    
    @property
    def z(self):
        return self.background['z']
    
    @property
    def rho_m(self):
        return np.array([self.background[f'(.)rho_{k}'] for k in ['b','cdm','ur']]).sum(axis=0)
    
    @property
    def rho_b(self):
        return self.background['(.)rho_b']   
    
    @property
    def rho_cdm(self):
        return self.background['(.)rho_cdm']    
    
    @property
    def rho_ur(self):
        return self.background['(.)rho_ur']    
    
    @property
    def rho_g(self):
        return self.background['(.)rho_g']
    
    @property
    def rho_de(self):
        return self.background[self.DE_id]
    
    @property
    def rho_crit(self):
        return self.background['(.)rho_crit']
    
    @property
    def background(self):
        return self._background()
    
    @property
    def ell(self):
        return self.Cls['ell']
        
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
    def Omega_cdm(self):
        return self.cosmo.Omega0_cdm()
    
    @property
    def Omega_k(self):
        return self.cosmo.Omega0_k()
    
    @property
    def DE_id(self):
        _DE_KEYS={'Modified Gravity':'(.)rho_smg','Scalar Field':'(.)rho_scf',
          'Cosmological Constant': '(.)rho_lambda','Fluid':'(.)rho_fld'}
        return _DE_KEYS[self.DE_type]
    
    @property
    def Omega_DE(self):
        if self.DE_type=='Cosmological Constant':
            return self.cosmo.Omega_Lambda
        return self.background[self.DE_id][-1]
    
    @property
    def rdrag(self):
        return self.background['comov.snd.hrz.']
    
    @property
    def _is_fluid(self)->bool:
        if 'Omega_smg' in self.info.keys():
            return (self.info['Omega_Lambda']==0 and self.info['Omega_smg']==0)
        else:
            return (self.info['Omega_Lambda']==0 and self.info['Omega_scf']==0)
    
    @property
    def _is_scf(self)->bool:
        if 'Omega_smg' in self.info.keys():
            return False
        else:
            return (self.info['Omega_Lambda']==0 and self.info['Omega_fld']==0)

    @property
    def _is_smg(self)->bool:
        if 'Omega_smg' in self.info.keys():
            return (self.info['Omega_Lambda']==0 and self.info['Omega_fld']==0)
        else:
            return False
        
    @property
    def DE_type(self):
        return self._get_DE_type()  
                
    def _get_DE_type(self)->str:
        DE_type='Cosmological Constant'
        if self._is_fluid:
            DE_type='Fluid'
        elif self._is_smg:
            DE_type='Modified Gravity'
        elif self._is_scf:
            DE_type='Scalar Field'        
        return DE_type      
    
def get_classy(info:dict,other_info:dict|None=None,verbose=0):
    """Get an instance of the Class class and compute observables requested in the `info` dictionary.

    Args:
        info (dict): common settings in a dictionary format
        other_info (dict | None, optional): Additional run-specific/precision settings. Defaults to None.

    Returns:
        Class instance: a Classy object with computations stored in it.
    """
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
        dict: a dictionary with the requested Cl's (2 < l <= l_max), in the requested units.
    """
    # Set the normalization to Firas T_0 measurements
    norm = T0_FIRAS**2 if units=='muK2' else 1.
    
    # Retrieve Cl's from Class
    Cls=cosmo.raw_cl() if lensed else cosmo.raw_cl()
    
    # Remove first two multipoles and scale accordingly
    l=Cls['ell'][2:]
    if ell_factor:
        l_factor = {key: l*(l+1) / TWO_PI for key in ['tt','te','et','ee','bb']}
        l_factor['pp']= (l*(l+1))**2 / TWO_PI
        l_factor['ell']= 1.
        l_factor.update({key: (l*(l+1))**(3/2) / TWO_PI for key in ['tp','pt','ep','pe']})
    else:
        l_factor={key:1 for key in ['tt','te','et','ee','tp','pt','ep','pe','pp','bb']}
        
    return  {key: norm * l_factor[key] * val[2:] for key,val in Cls.items()}

def get_alphas(cosmo) -> dict:
    """Get the evolution of the alpha functions from hiclass

    Args:
        cosmo (_type_): An instance of the hi_class/mochi_class class
    Returns:
        dict: a dictionary containing the evolution of the alpha functions
    """    
    # Retrieve alpha's from Class
    b = cosmo.get_background()
    alphas = {k: b[key] for k,key in zip(['M','B','K','T'],['Mpl_running_smg','braiding_smg','kineticity_smg','tensor_excess_smg'])}
    alpha_H = b['beyond_horndeski_smg'] if 'beyond_horndeski_smg' in b.keys() else np.zeros_like(b['Mpl_running_smg'])
    alphas.update({'H':alpha_H})
    return alphas

if __name__=='__main__':
    from classy import Class
    
    settings={'output':'tCl,pCl,lCl,mPk','lensing':'yes'}
    
    dic=Class()
    ini=Class()
    yaml=Class()
    # param=Class()
    
    info_ini='/Users/rodrigocalderon/Documents/Cosmology/AxiCLASS-master/base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini'
    info_yaml=''
    info_dic=settings
    infos=[info_ini]#,info_yaml,info_dic]
    for m,info in zip([ini,yaml,dic],infos):
        cosmo=ClassEngine(info=info,cosmo=m,name='name')
        print(cosmo.Cls)


