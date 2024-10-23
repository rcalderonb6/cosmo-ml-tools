from typing import Dict, Union
from jax import jit
import jax.numpy as jnp

@jit
def ExpontentialSquaredKernel(X: jnp.ndarray, Y: jnp.ndarray,
              params: Dict[str, jnp.ndarray],
              noise: int = 0, **kwargs: float) -> jnp.ndarray:
    """
    A Jax implementation of the ExponentialSquared Kernel.

    Args:
        X (jnp.ndarray): a jax array with X values
        Y (jnp.ndarray): a jax array with X values
        params (Dict[str, jnp.ndarray]): a dictionary with the kernel hyperparameter values. `params` should contain the keywords ['sigma_f','ell_f']
        noise (int, optional): additional white noise to be added to the diagonal of the covariance matrix. Defaults to 0.

    Returns:
        jnp.ndarray: Gaussian kernel evaluated at the X,Y values.
    """
    r2 = square_scaled_distance(X, Y, params["ell_f"])
    k = params["sigma_f"] * jnp.exp(-0.5 * r2)
    if X.shape == Y.shape:
        k += add_jitter(noise, **kwargs) * jnp.eye(X.shape[0])
    return k


@jit
def MaternKernel(X: jnp.ndarray, Y: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 noise: int = 0, **kwargs: float) -> jnp.ndarray:
    """
    A Jax implementation of the Matern Kernel.

    Args:
        X (jnp.ndarray): a jax array with X values
        Y (jnp.ndarray): a jax array with X values
        params (Dict[str, jnp.ndarray]): a dictionary with the kernel hyperparameter values. `params` should contain the keywords ['sigma_f','ell_f']
        noise (int, optional): additional white noise to be added to the diagonal of the covariance matrix. Defaults to 0.

    Returns:
        jnp.ndarray: Matern kernel evaluated at the X,Y values.
    """
    r2 = square_scaled_distance(X, Y, params["ell_f"])
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = params["sigma_f"] * (1 + sqrt5_r + (5/3) * r2) * jnp.exp(-sqrt5_r)
    if X.shape == Y.shape:
        k += add_jitter(noise, **kwargs) * jnp.eye(X.shape[0])
    return k

def _sqrt(x: jnp.ndarray, eps=1e-12)-> jnp.ndarray:
    """Helper function to ensure numerical stability

    Args:
        x (jnp.ndarray): an array with x values
        eps (float, optional): add tiny epsilon to ensure numerical stability. Defaults to 1e-12.

    Returns:
        jnp.ndarray: an array with sqrt(x + eps) values
    """
    return jnp.sqrt(x + eps)

def add_jitter(x: jnp.ndarray, jitter:float = 1e-6) -> jnp.ndarray:
    """Helper function to ensure numerical stability

    Args:
        x (jnp.ndarray): an array with x values
        jitter (float, optional): add tiny epsilon to ensure numerical stability. Defaults to 1e-12.

    Returns:
        jnp.ndarray: an array with x + jitter values
    """
    return x + jitter

def square_scaled_distance(X: jnp.ndarray, Y: jnp.ndarray,
                           lengthscale: Union[jnp.ndarray, float] = 1.
                           ) -> jnp.ndarray:
    """Helper function to compute the Eucledian distance between two points (arrays) X and Y.

    Args:
        X (jnp.ndarray): array of X values
        Y (jnp.ndarray): array of Y values
        lengthscale (Union[jnp.ndarray, float], optional): lengthscale of the kernel (typically ell_f). Defaults to 1.

    Returns:
        jnp.ndarray: the squared distance between two arrays X,Y
    """
    scaled_X = X / lengthscale
    scaled_Y = Y / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Y2 = (scaled_Y ** 2).sum(1, keepdims=True)
    XY = jnp.matmul(scaled_X, scaled_Y.T)
    r2 = X2 - 2 * XY + Y2.T
    return r2.clip(0)
