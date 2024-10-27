from .ensemble import EnsembleBase
from cobaya.model import get_model
import pocomc as pc
from scipy.stats import uniform,norm

class PocoMCBase(EnsembleBase):
    """
    PocoMC Base Class
    """
    def __init__(self,ini_file:str,engine:str='pocomc',sampler_kwargs:dict|None=None):
        # super().__init__(ini_file,engine,sampler_kwargs)
        self._priors=get_priors_from_cobaya(self.info)
        self._vectorized=True if 'vectorize' in self.sampler_kwargs.keys() else False
        
        # PocoMC Sampler
        self.sampler=pc.Sampler(prior=self.priors,
                                likelihood=self.log_likelihood,
                                **self.sampler_kwargs)
        
    def log_likelihood(self,*args,**kw_args):
        pass
        # raise NotImplementedError
    
    @property
    def logZ(self):
        logz, logz_err = self.sampler.evidence()
        return logz
    
    @property
    def samples(self):
        """ 
        MCMC samples with unit weights
        """
        samples, _, _ = self.sampler.posterior(resample=True)
        return samples
    

class PocoMCobaya(PocoMCBase):
        
    def __init__(self,ini_file:str,engine:str='pocomc',sampler_kwargs:dict|None=None):
        # super().__init__(ini_file,engine,sampler_kwargs)
        self.model=get_model(self.info)
        
    def log_likelihood(self,theta):
        if self._vectorized: # Unelegant solution when setting vectorize=True in PocoMC
            return [self.model.loglike(p,make_finite=True,return_derived=False) for p in theta]
        return self.model.loglike(theta,make_finite=True,return_derived=False)
        
def get_priors_from_cobaya(info:dict):
    """
    Generate Priors in PocoMC format from a cobaya-like info dictionary

    Args:
        info (dict): priors and settings for the run

    Returns:
        Prior: an instance of the PocoMC.Prior class
    """
    priors=[]
    for parameter,settings in info['params'].items():
        prior=settings['prior']
        if 'norm' in prior:
            loc,scale=[prior['norm'][key] for key in ['loc','scale']]
            priors.append(norm(loc,scale))
        else:
            lower_bound=prior['min']
            width=prior['max']-lower_bound
            priors.append(uniform(lower_bound,width))
    
    return pc.Prior(priors)