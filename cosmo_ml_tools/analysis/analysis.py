from .base import AnalysisBase
from getdist import loadMCSamples
    
class Analysis(AnalysisBase):
    
    def load(self):
        self._chains=load_chains(self.chains,self.labels)
        
    def add_chain(self, chain: str, label: str=None, root: str='',index:int=-1):
        if label is None: 
            label='chain0'
            self._chains.update(load_chains([chain],[label],root=root))
        else:
            pass

    def computeEvidence(self,chain:str=None) -> None:
        pass
    
    
    def load(self,chains:list[str]) -> None:
        pass   
    
    
    def add_chain(chain:str,label:str,root:str,index:int=-1):
        pass    
    
    def plot_triangle(self,params:list[str]=None):
        pass
    
    
    def plot_2D(self,params:list[str]):
        pass
    
    def plot_posterior_y(self,x,f,theta):
        pass
    
    def _getGelmanRubin(self):
        pass
    
    def getInfo(self):
        pass
    
    def set_aliases(self,aliases:dict) -> None:
        pass    
            
def load_chains(chains: list, labels: list, root: str='') -> dict:
    return {lbl: loadMCSamples(root+chain_fn) for lbl,chain_fn in zip(labels,chains)}
