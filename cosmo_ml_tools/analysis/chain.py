import numpy as np
from .base import ChainBase
from getdist import loadMCSamples

class MHChain(ChainBase):    
    """
    Metropolis-Hastings Base Class
    """
    def load(self,engine:str='getdist'):
        if engine=='getdist':
            return loadMCSamples(self._root+self.fn,self.gd_settings)
        else:
            raise NotImplementedError
    
    def to_harmonic(self):
        return NotImplementedError
    
    def getInfo(self):
        return NotImplementedError    
    
    def getTable(self, params: list):
        return NotImplementedError
    
    def set_param_labels(self,labels:list[str]):
        return NotImplementedError

class NSChain(ChainBase):
    """
    Nested-Sampling Base Class.
    """
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

def convert_to_harmonic(chain_fn:str,ndim:int,N:int=4,sampler:str='cobaya',ignore:float=0.3)-> tuple[list,list]:
    """Helper functionn to convert a set of chains into a Harmonic-friendly format

    Args:
        chain_fn (str): the location of the chains on the disk, where chain_fn is the root for all the chains (and .param_names files)
        ndim (int): Number of sampled (free parameters)
        N (int, optional): Number of chains {chain_fn}__i.txt with i from 1 to N. Defaults to 4.
        sampler (str, optional): specifies sampler used to compute the samples, useful for the format. Defaults to 'cobaya'.
        ignore (float, optional): The fraction of samples to reject as burn-in. Defaults to 0.3.

    Returns:
        tuple[list,list]: a tuple with the samples and log-posterior values in a Harmonic-compatible format
    """
    #Load individual chains
    chains={f'chain{i}': np.loadtxt(f'{chain_fn}.{i}.txt') for i in range(1,N)}
    
    # Determine the smaller of them and determine burn-in
    min_len=np.min([chain.shape[0] for chain in chains.values()])
    burn_in=int(ignore * min_len)
    
    # Reshape them into harmonic-friendly format
    if sampler=='cobaya':
        samples=np.array([chain[burn_in:min_len,2:ndim+2] for chain in chains.values()]).reshape((N,min_len-burn_in,ndim))
        lnprob=-np.array([chain[burn_in:min_len,1] for chain in chains.values()]).reshape((N,min_len-burn_in))
    
    #TODO: Implement Montepython compatibility (Should check which columns give lnprob and whether the sampled params are first)
    elif sampler=='montepython':
        print('Sorry, Montepython not yet implemented!')
        return
   
    else:
        print('Sorry, sampler not recognized or not yet implemented!')
        return
    
    return samples, lnprob