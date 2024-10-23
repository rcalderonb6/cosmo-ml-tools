from abc import ABC, abstractmethod

class MCBase(ABC):
    """Abstract Monte Carlo Base Class"""
    def __init__(self):
        ...
    
    @abstractmethod
    def run(self,*args,**kwargs):
        """
        Start the sampling process
        """
        ...
    
    @abstractmethod 
    def resume(self,*args,**kwargs):
        """
        Resume the sampling process
        """
        ...
            
    @abstractmethod
    def trace(self,params:list,*args,**kwargs):
        """
        Produce a trace plot of the chain. i.e. the evolution of the parameter values as a function of step number
        
        Args:
            params (list): A list of parameters to plot.
        
        Returns:
            fig: An instance of the matplotlib class
        """
        ...