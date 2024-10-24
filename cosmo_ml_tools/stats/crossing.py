import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from tqdm import tqdm

def get_Chebyshev_T(x:float|np.ndarray,Ci:list|np.ndarray):
    """
    Get a Chebyshev Polynomial expansion for a given set of coefficients.

    Args:
        x (float | np.ndarray): an array of values where to compute
        Ci (list | np.ndarray): list or array with Chebyshev coefficients

    Returns:
        callable: A callable to be evualuated at points `x`
    """
    cheb = Chebyshev(Ci,domain=(x[0],x[-1]))
    return cheb

def analytical_fde_from_w(z,C0=1,C1=0,C2=0,C3=0,C4=0,C5=0,C6=0):
    """Compute the integral fde=exp[\int 3(1+w) dln(1+z)] for a Chebyshev expansion of w(z) up to order 7 (C6)

    Args:
        z (array): redshift array
        C0 (int, optional): First Chebyshev coefficient. Defaults to 1.
        C1 (int, optional): Second Chebyshev coefficient. Defaults to 0.
        C2 (int, optional): Third Chebyshev coefficient. Defaults to 0.
        C3 (int, optional): Fourth Chebyshev coefficient. Defaults to 0.
        C4 (int, optional): Fifth Chebyshev coefficient. Defaults to 0.
        C5 (int, optional): Sixth Chebyshev coefficient. Defaults to 0.
        C6 (int, optional): Seventh Chebyshev coefficient. Defaults to 0.

    Returns:
        array: corresponding dark energy density evolution fde(z).
    """
    zmax=z.max()
    return (1 + z)**((3*zmax*(-128*C4*zmax + zmax**2*(-256*C4 - 8*(C2 + 20*C4)*zmax + 2*(C1 - 4*(C2 + 4*C4))*zmax**2 - (-1 + C0 - C1 + C2 + C4)*zmax**3 + C3*(2 + zmax)*(16 + zmax*(16 + zmax))) + C5*(2 + zmax)*(256 + zmax*(4 + zmax)*(128 + zmax*(44 + zmax)))) - 3*C6*(8 + zmax*(8 + zmax))*(256 + zmax*(512 + zmax*(320 + zmax*(64 + zmax)))))/zmax**6)/np.exp((2*z*(256*C6*(-60 + z*(30 + z*(-20 + z*(15 + 2*z*(-6 + 5*z))))) + 64*(C5 - 12*C6)*(60 + z*(-30 + z*(20 + 3*z*(-5 + 4*z))))*zmax + 80*(C4 - 10*C5 + 54*C6)*(-12 + z*(6 + z*(-4 + 3*z)))*zmax**2 + 40*(C3 - 8*C4 + 35*C5 - 112*C6)*(6 + z*(-3 + 2*z))*zmax**3 + 30*(C2 - 6*C3 + 20*C4 - 50*C5 + 105*C6)*(-2 + z)*zmax**4 + 15*(C1 - 4*C2 + 9*C3 - 16*C4 + 25*C5 - 36*C6)*zmax**5))/(5.*zmax**6))

def get_samples_crossing_fde(z:np.ndarray,samples_gd:dict,order:int=4) -> dict:
    """Get the dark energy evolution from MCMC samples of the Chebyshev coefficients

    Args:
        z (np.ndarray): array with redshift values
        samples_gd (dict): a dictionary with getdist instances. They keys in the dictionary are used as labels for each chains.
        order (int, optional): The order of the polynomial expansion of fde(z)=\rho_{\rm DE}(). Defaults to 4.

    Returns:
        dict: a dictionary containing the corresponding samples of fde(z) for each chain.
    """
    coeffs=[f'C{i}' for i in range(order)]
    samples_hyper={label: np.array([samples[c] for c in coeffs]).T for label,samples in samples_gd.items()}
    crossing_fde={label: np.array([get_Chebyshev_T(z,Ci)(z) for Ci in tqdm(samples)]) for label,samples in samples_hyper.items()}
    return crossing_fde

def get_samples_crossing_w(z,samples_gd:dict,order:int=4,return_fde:bool=True) -> dict:
    """Get the dark energy evolution 
    ..math::
        f_{\rm DE}(z) = \rho_{\rm DE}(z)/\rho_{\rm DE,0}
        for a given set of Chebyshev coefficients in the expansion of 
    ..math::
        w(z)=\sum_i^Nc_i T_i(x)

    Args:
        z (_type_): _description_
        samples_gd (dict): _description_
        order (int, optional): _description_. Defaults to 4.
        return_fde (bool, optional): _description_. Defaults to True.

    Returns:
        dict: _description_
    """
    coeffs=[f'C{i}' for i in range(order)]
    samples_hyper={label: np.array([samples[c] for c in coeffs]).T for label,samples in samples_gd.items()}
    crossing_w={label: np.array([-get_Chebyshev_T(z,Ci)(z) for Ci in tqdm(samples)]) for label,samples in samples_hyper.items()}
    crossing_fde={lbl: np.array([analytical_fde_from_w(z,*Ci) for Ci in tqdm(samples)]) for lbl,samples in samples_hyper.items()}
    if return_fde:
        return crossing_w,crossing_fde
    return crossing_w