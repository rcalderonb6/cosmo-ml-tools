from abc import ABC, abstractmethod

class ChainBase(ABC):
    """
    Abstract Base Chain Class
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def load(self):
        """Load the chain using the engine of your choice"""
        ...
    
    @abstractmethod
    def to_harmonic(self):
        """
        A method that returns the samples and corresponding posterior values in a Harmonic-friendly format.
        """
        ...
    
    @abstractmethod
    def set_param_labels(self,labels:list[str]):
        """Set the labels for the parameters"""
        ...
        
    @abstractmethod
    def set_label(self):
        """Set the label for the chain"""
        ...
    
    @abstractmethod
    def set_alias(self):
        """
        Set an alias, or shorter name of the chain for quickly reference (useful for long data combinations)
        """
        ...
    
    @abstractmethod
    def getInfo(self):
        """
        Print a summary of useful information on the chain.
        """
        ...
    
    @abstractmethod
    def getTable(self,params:list):
        """
        Print a latex table with constraints.
        """
        ...
        
class AnalysisBase(ABC):
    """
    Abstract Analysis Class
    """
    def __init__(self) -> None:
        pass
    
    def set_labels(self,labels) -> None:
        self._labels=labels
        
    @abstractmethod
    def computeEvidence(self):
        ...
        
    @abstractmethod
    def load(self,chains):
        ...    
    
    @abstractmethod
    def add_chain(chain:str,label:str,root:str):
        ...
        
    @abstractmethod
    def plot_triangle(self,params:list[str]=None):
        ...    
    
    @abstractmethod
    def plot_2D(self):
        ...    
    
    @abstractmethod
    def plot_posterior_y(self,x,f,theta):
        ...    
    
    @abstractmethod
    def _getGelmanRubin(self):
        ...   
    
    @abstractmethod
    def getInfo(self):
        ...    
    
    @abstractmethod
    def set_aliases(self,aliases):
        ...        
        
    @property
    def GelmanRubin(self) -> list:
        self._getGelmanRubin(self)
        
    @property
    def chains(self) -> dict:
        return self._chains
    
    @property
    def labels(self) -> list:
        return self._labels
    
    @property
    def filenames(self)-> list:
        return self._filenames
    
    @property
    def N(self) -> int:
        return len(self._labels)
