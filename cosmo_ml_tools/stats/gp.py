from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jra

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.enable_x64()

from .base import GaussianProcessBase

class GaussianProcessJax(GaussianProcessBase):
    def __init__(self, kernel, input_dim: int, mean_fn=None): 
        """
        Base Class implementing the usual Gaussian Process Regression algorithm. 
        The GP posterior is sampled using the Hamiltonian Monte Carlo 'No-U Turn' Sampler (NUTS) as implemented in numpyro
        e.g. BaseGP(input_dim=2, kernel=RBFKernel)
        """
        # clear_cache()
        self.input_dim = input_dim
        self.kernel = kernel
        self.mean_fn = mean_fn
        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X, y):
        """GP model"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        
        # Sample kernel parameters and noise
        with numpyro.plate('k_param', self.input_dim):  # allows using ARD kernel for input_dim > 1
            length = numpyro.sample("ell_f", dist.LogNormal(0.0, 1.0))
            scale = numpyro.sample("sigma_f", dist.LogNormal(0.0, 1.0))
            noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        
            # Add mean function (if any)
            if self.mean_fn is not None:
                f_loc += self.mean_fn(X).squeeze()
            
            # compute kernel
            k = self.kernel(
                X, X,
                {"ell_f": length, "sigma_f": scale},
                noise
            )
            # sample y according to the standard Gaussian process formula
            numpyro.sample(
                "y",
                dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
                obs=y,
            )

    def run_MCMC(self, rng_key, X, y,
            num_warmup=2000, num_samples=2000, num_chains=1,
            progress_bar=True, print_summary=True):
        """
        Run MCMC to infer the GP model parameters
        """
        X = X if X.ndim > 1 else X[:, None]
        self.X_train = X
        self.y_train = y

        init_strategy = numpyro.infer.init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            jit_model_args=False
        )
        self.mcmc.run(rng_key, X, y)
        if print_summary:
            self.mcmc.print_summary()
    
    def get_mcmc_samples(self, chain_dim=False):
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    @partial(jit, static_argnames='self')
    def get_posterior(self, X_test, params):
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP hyperparameters
        """
        noise = params["noise"]
        y_residual = self.y_train
        if self.mean_fn is not None:
            y_residual -= self.mean_fn(self.X_train).squeeze()
            # y_residual -= self.mean_fn(self.X_train, params).squeeze()
            
        # compute kernel matrices for train and test data
        k_pp = self.kernel(X_test, X_test, params, noise)
        k_pX = self.kernel(X_test, self.X_train, params, jitter=0.0)
        k_XX = self.kernel(self.X_train, self.X_train, params, noise)
        
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        
        if self.mean_fn is not None:
            mean += self.mean_fn(X_test).squeeze()
            # mean += self.mean_fn(X_test, params).squeeze()
        return mean, cov
        
    def _predict(self, rng_key, X_test, params, n):
        """Prediction with a single sample of GP hyperparameters"""
        X_test = X_test if X_test.ndim > 1 else X_test[:, None]
        
        # Get the predictive mean and covariance
        y_mean, K = self.get_posterior(X_test, params)
        
        # draw samples from the posterior predictive for a given set of hyperparameters
        y_sample = dist.MultivariateNormal(y_mean, K).sample(rng_key, sample_shape=(n,))
        
        return y_mean, y_sample.squeeze()
    
    def predict(self, rng_key, X_test, samples=None, n=1):
        """Make prediction at X_test points using sampled GP hyperparameters"""
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        num_samples = samples["ell_f"].shape[0]
        
        # use vmap for 'vectorization'
        vmap_args = (jra.split(rng_key, num_samples), samples)
        predictive = jax.vmap(lambda params: self._predict(params[0], X_test, params[1], n))
        
        y_means, y_sampled = predictive(vmap_args)
        
        return y_means.mean(0), y_sampled


# if jax.__version__ < '0.2.26':
#     clear_cache = jax.interpreters.xla._xla_callable.cache_clear
# else:
#     clear_cache = jax._src.dispatch._xla_callable.cache_clear
     
if __name__=='__main__':
    print('All Good')