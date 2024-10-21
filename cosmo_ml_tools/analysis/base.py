from abc import ABC, abstractmethod
try:
    from getdist import plots, loadMCSamples
except ModuleNotFoundError: 
    print ("Getdist is a requirement for this Class to work. \
        Consider installing Getdist by typing: python -m pip install getdist")

class ChainBase(ABC):
    """
    Abstract Chain Class
    """
    def __init__(self,fn,root='',label=None,
                 param_names:list=None,engine='getdist',
                 analysis_settings:dict={'ignore_rows':0.3}) -> None:
        
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
        ...
        
    @abstractmethod
    def set_label(self):
        """Set the label for the chain (in .tex format)"""
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
    def __init__(self,chains:list[str],
                 labels:list[str]=None,engine:str='getdist',root_dir='',
                 analysis_settings:dict={'ignore_rows':0.3}) -> None:
        
        self._filenames=chains
        self._root=root_dir
        self._engine=engine
        self._labels=[f'chain{i}' for i in range(len(chains))] if labels is None else labels 
        self._chains=self.load_chains(chains)
    
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
    
if __name__=='__main__':
    pass