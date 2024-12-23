from abc import ABC, abstractmethod

class BoltzmannBase(ABC):
    """
    Abstract Base Class for the Boltzmann Solver class
    """
    
    @property
    def name(self)->None:
        self._name
    
    @abstractmethod
    def getInfo(self) -> None:
        """
        Print general information about the current Cosmology
        """
        ...
        
    @abstractmethod
    def update(self,**kwargs) -> None:
        """
        Update the values of the cosmological parameters with the provided dictionary and recompute observables.
        """
        ...
    
    @abstractmethod
    def compute(self) -> None:
        """
        (Re)-compute the observables for the chosen cosmology
        """
        ...
        
    @abstractmethod
    def plot(self, observables: list[str],*args,**kwargs) -> None:
        """
        Plot one or more of the requested output observables

        Args:
            observables (list[str]): One of the output observables requested to Class. e.g. P(k), Cl's...
        """
        ...
        
    @abstractmethod
    def store(self,filename) -> None:
        """
        Store the requested outputs in a given folder
        """
        ...
        
    @abstractmethod
    def empty(self) -> None:
        """
        Clean up structure
        """
        ...
        