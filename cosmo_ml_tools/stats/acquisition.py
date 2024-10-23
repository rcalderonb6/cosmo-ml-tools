from typing import Dict, Type
import jax.numpy as jnp
import jax.random as jra
import numpyro.distributions as dist

from .gp import GaussianProcessJax

def ExpectedImprovement(rng_key: jnp.ndarray, model: Type[GaussianProcessJax],
       X: jnp.ndarray, xi: float = 0.01,
       maximize: bool = False, n: int = 1) -> jnp.ndarray:
    """
    Expected Improvement
    """
    y_mean, y_sampled = model.predict(rng_key, X, n=n)
    if n > 1:
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
    mean, sigma = y_sampled.mean(0), y_sampled.std(0)
    u = (mean - y_mean.max() - xi) / sigma
    u = -u if not maximize else u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    return sigma * (updf + u * ucdf) 


def UpperConfidenceBound(rng_key: jnp.ndarray, model: Type[GaussianProcessJax],
        X: jnp.ndarray, beta: float = .25,
        maximize: bool = False, n: int = 1) -> jnp.ndarray:
    """
    Upper confidence bound
    """
    _, y_sampled = model.predict(rng_key, X, n=n)
    if n > 1:
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
    mean, var= y_sampled.mean(0), y_sampled.var(0)
    delta = jnp.sqrt(beta * var)
    if maximize:
        return mean + delta
    return mean - delta 


def UncertaintyExploration(rng_key: jnp.ndarray,
       model: Type[GaussianProcessJax],
       X: jnp.ndarray, n: int = 1) -> jnp.ndarray:
    """Uncertainty-based exploration (aka kriging)"""
    _, y_sampled = model.predict(rng_key, X, n=n)
    return y_sampled.var(0)


def ThompsonSampling(rng_key: jnp.ndarray,
             model: Type[GaussianProcessJax],
             posterior_samples: Dict[str, jnp.ndarray],
             X: jnp.ndarray, n: int = 1) -> jnp.ndarray:
    """Thompson sampling"""
    posterior_samples = model.get_samples()
    idx = jra.randint(rng_key, (1,), 0, len(posterior_samples["k_length"]))
    samples = {k: v[idx] for (k, v) in posterior_samples.items()}
    _, tsample = model.predict(rng_key, X, samples, n)
    if n > 1:
        tsample = tsample.mean(1)
    return tsample.squeeze()
     
