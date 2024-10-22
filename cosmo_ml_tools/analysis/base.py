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
        Print a latex table with the mean and 68% CL.
        """
        ...
    
    @abstractmethod
    def trace(self):
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
    def computeEvidence(self,chain:str=None) -> None:
        """Compute the Bayesian Evidence for the chains."""
        ...
        
    @abstractmethod
    def load(self,chains:list[str]) -> None:
        """
        Load a given set of chains.
        """
        ...    
    
    @abstractmethod
    def add_chain(chain:str,label:str,root:str):
        ...
        
    @abstractmethod
    def plot_triangle(self,params:list[str]=None):
        """
        Triangle plot for the specified subset of parameters.
        """
        ...    
    
    @abstractmethod
    def plot_2D(self,params:list[str]):
        """
        Plot the 2D marginalized posteriors.
        """
        ...    
    
    @abstractmethod
    def plot_posterior_y(self,x,f,theta):
        """
        Plot the posterior distribution of a given function y = f(x,theta)
        from MCMC samples of theta using the ``f_given_x`` package.
        """
        ...    
    
    @abstractmethod
    def _getGelmanRubin(self):
        """Computee the Gelman-Rubin (R-1) statistics for the chains"""
        ...   
    
    @abstractmethod
    def getInfo(self):
        """
        Print useful information about the chains loaded.
        """
        ...    
    
    @abstractmethod
    def set_aliases(self,aliases:dict) -> None:
        """
        Set aliases for the chains
        """
        ...        
        
    @property
    def GelmanRubin(self) -> list:
        """
        Gelman Rubin (R-1) statistics for the chains loaded.
        """
        self._getGelmanRubin(self)
        
    @property
    def chains(self) -> dict:
        """
        The chains stored as a dictionary with labels as keys.
        """
        return self._chains
    
    @property
    def labels(self) -> list:
        """
        Labels for the chains that are loaded.
        """
        return self._labels
    
    @property
    def filenames(self)-> list:
        """
        List of filenames for the chains, as stored on the disk.
        """
        return self._filenames
    
    @property
    def N(self) -> int:
        """
        Number of chains loaded.
        """
        return len(self._labels)
