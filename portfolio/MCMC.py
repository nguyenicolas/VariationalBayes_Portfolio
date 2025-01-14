import jax
from jax import lax
from typing import Tuple
import jax.numpy as jnp
from portfolio.base import BayesianPortfolioConstructor
from tqdm.auto import tqdm
key = jax.random.PRNGKey(0)

class MCMC_Portfolio(BayesianPortfolioConstructor):
    """MCMC portfolio construction on Gaussian-Wishart model
    """

    def __init__(self, data: jnp.array, params: dict) -> None:
        super().__init__(data, params)
        self.delta = jnp.zeros(self.d)
        self.create_priors()
    
    def create_priors(self) -> None:
        return super().create_priors()
    
    def one_step_GD(self) :

        #step_size = self.compute_step_size()
        delta_old = jnp.copy(self.delta)
        grad_evaluated = self.gradient_estimate(delta_old)
        norm_grad = jnp.linalg.norm(grad_evaluated)
        print(norm_grad)

        delta_new = delta_old - self.step_size_GD * grad_evaluated
        self.delta = jnp.copy(delta_new)

    def construct(self):
        for _ in range(self.nb_GD_iter) :
            #if iter == self.nb_GD_iter - 1:
            #    self.evaluate = True
            self.one_step_GD()
        #delta = jnp.copy(self.delta) 
        #self.MCMC_evaluate["delta"] = delta

        return self.delta
    
    def compute_mean_parameters(self, Lambda):
        """Computes the distribution \pi(\mu\condi\Lambda)
        """

        # Conditional mean posterior
        mean_mu = jnp.dot(
            jnp.linalg.inv(Lambda + (1/self.n) * self.Lambda_0),
            jnp.dot(Lambda, jnp.mean(self.data, axis=0)) + (1/self.n) * jnp.dot(self.Lambda_0, self.mu_0)
        )
        variance_mu = (1/self.n) * jnp.linalg.inv(Lambda + (1/self.n)*self.Lambda_0)
        return mean_mu, variance_mu

    def compute_variance_parameters(self, mu):
        """Computes the distribution \pi(\Lambda\condi\mu)
        """
        # Conditional precision posterior
        nu_Lambda = self.n + self.nu_0
        mu_helper = jnp.tile(mu, self.n).reshape(self.n, self.d)
        psi_Lambda = jnp.linalg.inv(jnp.einsum('ij,ik->jk', self.data - mu_helper, self.data - mu_helper) + self.psi_0_inv)
        return nu_Lambda, psi_Lambda
    

    def gradient_estimate(self, delta):
        """
        Outputs M pairs (mu_k, Lambda_k)
        """
        
        def body_fn(k, carry):
            # Unpack the current state
            list_y, key, mu_k, Lambda_k = carry

            # Gibbs sampling
            mean_mu, variance_mu = self.compute_mean_parameters(Lambda_k)
            mu_k = jax.random.multivariate_normal(key, mean_mu, variance_mu)

            nu_Lambda, psi_Lambda = self.compute_variance_parameters(mu_k)
            Lambda_k = self.sample_wishart(key, nu_Lambda, psi_Lambda)
            Sigma_k = jnp.linalg.inv(Lambda_k)

            y_k = jax.random.multivariate_normal(key, mu_k - self.lambda_ * jnp.dot(Sigma_k, delta), Sigma_k)

            # Accumulate the result
            list_y = list_y.at[k].set(y_k)

            # Return the updated state
            return list_y, key, mu_k, Lambda_k

        # Initialize values for the loop
        list_y = jnp.zeros((self.M, self.d))  # We still need to initialize the shape
        key = jax.random.PRNGKey(0)  # Assuming `key` is provided elsewhere in your code
        mu_k, Lambda_k = jnp.ones(self.d), jnp.ones((self.d, self.d))

        # Use lax.fori_loop to accumulate `list_y`
        list_y, _, _, _ = lax.fori_loop(0, self.M, body_fn, (list_y, key, mu_k, Lambda_k))

        # Compute the gradient estimate
        gradient_estimate = - self.lambda_ * jnp.mean(list_y, axis=0)

        return gradient_estimate
    
    def sample_wishart(self, key, df, scale_matrix):
        """
        Samples from the Wishart distribution using the method of generating a sample covariance matrix.
        
        Args:
        - key: PRNG key for random number generation.
        - df: Degrees of freedom (must be >= d).
        - scale_matrix: A positive definite scale matrix of shape (d, d).
        
        Returns:
        - A sample from the Wishart distribution.
        """
        
        # Dimensionality of the scale matrix
        d = scale_matrix.shape[0]
        
        # Check if degrees of freedom is valid
        if df < d:
            raise ValueError("Degrees of freedom must be greater than or equal to the dimensionality d.")
        
        # Cholesky decomposition of the scale matrix to sample from a normal distribution
        L = jnp.linalg.cholesky(scale_matrix)  # Cholesky factorization
        
        # Generate `df` samples from a standard normal distribution (shape: [df, d])
        key, subkey = jax.random.split(key)
        normal_samples = jax.random.normal(subkey, shape=(df, d))
        
        # Multiply the normal samples by the Cholesky decomposition (i.e., perform a linear transformation)
        transformed_samples = jnp.dot(normal_samples, L.T)  # Shape: [df, d]
        
        # Compute the sample covariance matrix: (X^T X), where X is the matrix of transformed samples
        wishart_sample = jnp.dot(transformed_samples.T, transformed_samples)
        
        return wishart_sample






    """
    def gradient_estimate(self, delta):
        
        #ouputs M pairs (mu_k, Lambda_k)
        

        list_y = jnp.zeros((self.M, self.d))

        mu_k, Lambda_k = jnp.ones(self.d), jnp.ones((self.d, self.d))

        for k in range(self.M):

            # Gibbs sampling
            mean_mu, variance_mu = self.compute_mean_parameters(Lambda_k)
            mu_k = jax.random.multivariate_normal(key, mean_mu, variance_mu)

            nu_Lambda, psi_Lambda = self.compute_variance_parameters(mu_k)
            Lambda_k = self.sample_wishart(key, nu_Lambda, psi_Lambda)
            Sigma_k = jnp.linalg.inv(Lambda_k)

            y_k = jax.random.multivariate_normal(key, mu_k - self.lambda_ * jnp.dot(Sigma_k, delta), Sigma_k)
            list_y[k] = y_k

        gradient_estimate = - self.lambda_ * jnp.mean(list_y, axis=0)

        return gradient_estimate
    """