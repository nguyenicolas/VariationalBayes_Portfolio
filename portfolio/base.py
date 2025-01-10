import jax.numpy as jnp
from abc import ABC, abstractmethod

class PortfolioConstructor(ABC):
    """Abstract class for a VB-portfolio decision."""

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
    
    @abstractmethod
    def construct(self)-> jnp.array:
        """Returns decision vector.
        """
        pass