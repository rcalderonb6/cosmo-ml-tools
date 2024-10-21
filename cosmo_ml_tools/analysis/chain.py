import numpy as np
from base import ChainBase
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

def convert_to_harmonic(fn,ndim,N=4,sampler='cobaya'):
    
    #Load individual chains
    chains={f'chain{i}': np.loadtxt(f'{fn}.{i}.txt') for i in range(1,N)}
    
    # Determine the smaller of them and remove burning
    min_len=np.min([chain.shape[0] for chain in chains.values()])
    burnin=min_len//4
    
    # Reshape them into harmonic-friendly format
    if sampler=='cobaya':
        samples=np.array([chain[burnin:min_len,2:ndim+2] for chain in chains.values()]).reshape((N,min_len-burnin,ndim))
        lnprob=-np.array([chain[burnin:min_len,1] for chain in chains.values()]).reshape((N,min_len-burnin))
    elif sampler=='montepython':
        print('Sorry, Montepython not yet implemented!')
        return
    else:
        print('Sorry, sampler not yet recognized!')
        return
    
    return samples, lnprob