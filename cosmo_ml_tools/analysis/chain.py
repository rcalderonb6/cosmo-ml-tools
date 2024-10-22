from .base import ChainBase
class MHChain(ChainBase):    
    """
    Metropolis-Hastings Base Class
    """
    def load(self):
        return NotImplementedError
    
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
