from .base import LikelihoodBase

class Likelihood(LikelihoodBase):
    """General Likelihood Class"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def ln_prob(self,*args,**kwargs):
        return -0.5 * self.chi(*args,**kwargs)