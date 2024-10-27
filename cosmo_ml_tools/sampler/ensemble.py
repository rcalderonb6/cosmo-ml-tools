from .base import MCBase
from utils.file import _initialize_helper

class EnsembleBase(MCBase):
    """
    Base Class for an Ensemble Sampler
    """
    def __init__(self,ini_file:str|dict,engine:str,sampler_kwargs:dict):
        # super().__init__()
        self.info=self._initialize_helper(ini_file)
        self.sampler_kwargs=sampler_kwargs
        self.out_dir=self.info['output'] if 'output' in self.info.keys() else None
        self._priors=None 
        
    def run(self):
        self.sampler.run()
        
    @property
    def priors(self):
        return self._priors