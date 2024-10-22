import numpy as np
from base import ChainBase
from utils.table import get_latex_table
from getdist import loadMCSamples

class MHChain(ChainBase):    
    """
    Metropolis-Hastings Base Class
    """
    def load(self):
        return loadMCSamples(self._root+self.fn,self.gd_settings)
    
    def to_harmonic(self):
        return NotImplementedError
    
    def getInfo(self):
        return NotImplementedError    
    
    def getTable(self, params: list):
        return NotImplementedError
    
    def set_param_labels(self,labels:list[str]):
        return NotImplementedError

class NSChain(ChainBase):
    pass    

class EMCEEChain(ChainBase):
    pass

class CobayaChain(MHChain):
    
    def to_harmonic(self,ndim:int,N:int):
        return convert_to_harmonic(self.filename,ndim,N=N,sampler='cobaya')

class MontePythonChain(MHChain):
    
    def to_harmonic(self,ndim:int,N:int):
        return convert_to_harmonic(self.filename,ndim,N=N,sampler='montepython')


#####################    
# Helpers functions    
#####################    

def convert_to_harmonic(chain_fn,ndim,N:int=4,sampler:str='cobaya',ignore:float=0.3)-> tuple[list,list]:
    
    #Load individual chains
    chains={f'chain{i}': np.loadtxt(f'{chain_fn}.{i}.txt') for i in range(1,N)}
    
    # Determine the smaller of them and determine burn-in
    min_len=np.min([chain.shape[0] for chain in chains.values()])
    burnin=int(ignore * min_len)
    
    # Reshape them into harmonic-friendly format
    if sampler=='cobaya':
        samples=np.array([chain[burnin:min_len,2:ndim+2] for chain in chains.values()]).reshape((N,min_len-burnin,ndim))
        lnprob=-np.array([chain[burnin:min_len,1] for chain in chains.values()]).reshape((N,min_len-burnin))
    
    #TODO: Implement Montepython compatibility
    elif sampler=='montepython':
        print('Sorry, Montepython not yet implemented!')
        return
   
    else:
        print('Sorry, sampler not recognized or not yet implemented!')
        return
    
    return samples, lnprob