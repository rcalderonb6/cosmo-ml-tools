from abc import ABC, abstractmethod
class ChainBase(ABC):
    """
    Abstract Base Chain Class
    """
    def __init__(self,fn,root='',label:str = None,
                 param_names:list = None, engine:str = 'getdist',
                 analysis_settings:dict = {'ignore_rows':0.3}) -> None:
        
        self._engine=engine
        self._root=root
        self.filename=fn
        self.param_names=param_names
        self.label='chain' if label is None else label
        self.alias=None
        self.gd_settings=analysis_settings
    
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
        
