import jax.numpy as jnp
from abc import ABC, abstractmethod

class BayesianPortfolioConstructor(ABC):
    """Abstract class for a portfolio decision that use a Bayesian prior."""

    def __init__(self, data: jnp.array, params: dict) -> None:
        """Initialize portfolio constructor.

        Args:
            data: jnp.array of size (n, d) where n is the sample size and d the number of assets.
            params: dict containing parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)

        self.data = data
        self.n, self.d = jnp.shape(self.data)
        self.delta = jnp.zeros(self.d) # initialize decision
        self.create_priors()
    
    @abstractmethod
    def create_priors(self) -> None:
       """Set prior parameters for the Gaussian-Wishart model.
        """
       self.mu_0 = jnp.mean(self.data, axis=0)
       self.Lambda_0 = jnp.eye(self.d)
       self.nu_0 = self.n + self.d
       cov = jnp.cov(self.data.T)
       self.psi_0_inv = cov
       self.psi_0 = jnp.linalg.inv(cov) 

    @abstractmethod
    def construct(self)-> jnp.array:
        """Returns decision vector.
        """
        pass