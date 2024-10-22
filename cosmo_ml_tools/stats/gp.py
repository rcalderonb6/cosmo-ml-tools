import numpy as np
from abc import ABC, abstractmethod

def GaussianProcess(ABC):
    """Gaussian Process Abstract Base class"""

    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self):
        ...