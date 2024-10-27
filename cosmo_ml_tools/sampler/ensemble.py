from .base import MCBase
from ..utils.file import initialize_helper

class EnsembleBase(MCBase):
    """
    Base Class for an Ensemble Sampler.
    """
    # def __init__(self,ini_file,engine:str,sampler_kwargs:dict):
    #     # super().__init__(ini_file,engine,sampler_kwargs)
    #     self.info=_initialize_helper(ini_file)
    #     self.sampler_kwargs=sampler_kwargs
    #     self.out_dir=self.info['output'] if 'output' in self.info.keys() else None
    #     self._priors=None 
        
    def run(self):
        self.sampler.run()
        
    @property
    def priors(self):
        return self._priors