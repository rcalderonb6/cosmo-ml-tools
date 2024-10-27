import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

def get_bestfit(samples_gd,params):
    """Get the best-fit chi2 values and corresponding parameter values from the chains.

    Args:
        samples_gd (dict): a dictionnary containing the GetDist samples with labels as keys
        params (list): a list of requested cosmological parameters

    Returns:
        tuple: chi2 values and corresponding dictionnary with best-fit parameters
    """
    chi2_min={label: samples['chi2'].min() for label,samples in samples_gd.items()}
    bf_params={label: np.array([samples[p][samples['chi2']==chi2_min[label]][0] for p in params]) for label,samples in samples_gd.items()}
    return chi2_min,bf_params

def get_w(z,fde):
    """Compute the equation of state from the dark energy density evolution

    Args:
        z (array): redshift array
        fde (array): array with the dark energy evolution f_DE(z)

    Returns:
        array: equation of state as a function of redshift, w(z)
    """
    return (1+z)/fde * np.gradient(fde,z)/3 - 1

def get_h(z,Om0,fde=None):
    """Compute the normalized expansion history H(z)/H0 from the dark energy density evolution fde(z) and fractional matter density Om0

    Args:
        z (array): redshift array
        Om0 (float|array): fractional matter density today, Omega_m
        fde (array): Optional, array with the dark energy evolution f_DE(z)

    Returns:
        array: equation of state as a function of redshift, w(z)
    """
    if fde is None:
        fde=np.ones_like(z)
    return np.sqrt(Om0 * (1+z)**3 + (1-Om0)*fde)

def get_OmegaDE(Om0,h,fde=None):
    """Compute the fractional energy density of dark energy, normalized to the critical one.

    Args:
        Om0 (float|array): values for the fractional matter density today
        h (array): array with the normalized expansion history
        fde (array): optional, array with the dark energy evolution f_DE(z)
        

    Returns:
        array: array with the dark energy density evolution, normalized to the critical density, i.e. Omega_DE(z)
    """
    if fde is None:
        fde=np.ones_like(h)
    return (1-Om0)*fde/h**2
    
def get_q(z,h):
    """Compute the deacceleration parameter for a given expansion history

    Args:
        z (array): redshift array
        h (array): array with the expansion history h(z)

    Returns:
        array: deceleration parameter as a function of z
    """
    return np.gradient(h,z)*(1+z)/h - 1

def get_q_from_samples_h(z,samples_h):
    hprime=np.gradient(samples_h,z,axis=1)
    return (1+z)/samples_h * hprime -1 

def get_Omz(z,h):
    num = h[:,1:]**2 - 1
    den = (1+z[1:])**3 - 1
    return num/den

def get_H_from_fde(z,fde,Om0,H0=1):
    return H0 * np.sqrt(Om0*(1+z)**3+(1-Om0)*fde)
