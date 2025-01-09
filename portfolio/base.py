import jax.numpy as jnp
from abc import ABC, abstractmethod

class PortfolioConstructor(ABC):
    def __init__(self, data: jnp, params: dict) -> None:
        """
        Initialize general portfolio constructor.

        Args:
            data (jnp.array) :
            lambda_ (float): risk aversion parameter (> 0)
        Raises:
            ValueError: If lambda_ is not positive
        """
        #if lambda_ <= 0:
        #    raise ValueError("Risk aversion must be positive.")
        
        for key, value in params.items():
            setattr(self, key, value)

        self.data = data
        self.n, self.d = jnp.shape(self.data)
    
    @abstractmethod
    def construct(self)-> jnp.array:
        """
        Abstract method to construct the portfolio.
        Subclasses must implement this method.

        Returns:
            np.ndarray: Portfolio weights.
        """
        pass

