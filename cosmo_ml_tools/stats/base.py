from abc import ABC, abstractmethod


class GaussianProcessBase(ABC):
    """Gaussian Process Abstract Base class"""

    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self):
        ...
    
    @abstractmethod
    def predict(self,*args,**kwargs):
        ...

class LikelihoodBase(ABC):
    """Likelihood Abstract Base Class"""
    def __init__(self,*args,**kwargs):
        pass