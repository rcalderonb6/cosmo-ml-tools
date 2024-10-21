from base import AnalysisBase
from getdist import loadMCSamples
    
def load_chains(chains: list, labels: list, root: str='') -> dict:
    return {lbl: loadMCSamples(root+chain_fn) for lbl,chain_fn in zip(labels,chains)}

class Analysis(AnalysisBase):
    
    def load(self):
        self._chains=load_chains(self.chains,self.labels)
        
    def add_chain(self, chain: str, label: str=None, root: str=''):
        if label is None: label='chain0'
        self._chains.update(load_chains([chain],[label],root=root))