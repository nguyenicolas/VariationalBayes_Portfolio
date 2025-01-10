from portfolio.base import PortfolioConstructor
import jax.numpy as jnp

class Markowitz(PortfolioConstructor):
    def __init__(self, data: jnp.array, params: dict) -> None:
        super().__init__(data, params)
    
    def construct(self) -> jnp.array:
        cov_estimate = self.compute_cov()
        mean_estimate = self.compute_mean()
        return jnp.dot(jnp.linalg.inv(cov_estimate), mean_estimate)
    
    def compute_cov(self) -> jnp.array:
        """Compute covariance estimate based on historical data.
        
        Returns:
            jnp.array of size (d, d).
        """
        return jnp.cov(self.data.T)
    def compute_mean(self) -> jnp.array:
        return jnp.mean(self.data, axis=0)