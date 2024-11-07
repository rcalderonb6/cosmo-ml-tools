
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class DefaultPrecision:
    def to_dict(self):
        return asdict(self)

@dataclass
class FeaturesPrecision(DefaultPrecision):
    k_per_decade_for_pk:int = 200
    k_per_decade_for_bao:int = 200

@dataclass
class EFTofDEPrecision(DefaultPrecision):
    non_linear_min_k_max:int=20
    accurate_lensing:int=1
    delta_l_max:int=1e3
    perturbations_sampling_stepsize:float=1e-2
    l_logstep:float=1.026
    k_per_decade_for_pk:int=200
    k_max_tau0_over_l_max:int=8



if __name__=='__main__':
    for p in [DefaultPrecision(),EFTofDEPrecision(),FeaturesPrecision()]:
        print(p,'\n\n')
        print(p.to_dict())
        print('-------------------------------------')