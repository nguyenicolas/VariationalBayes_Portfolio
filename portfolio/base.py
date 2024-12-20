import jax.numpy as jnp
from abc import ABC, abstractmethod

class PortfolioConstructor(ABC):
    def __init__(self, data: jnp, lambda_: float) -> None:
        """
        Initialize generall portfolio constructor.

        Args:
            data (jnp) :
            lambda_ (float): risk aversion parameter (> 0)
        Raises:
            ValueError: If lambda_ is not positive
        """
        if lambda_ <= 0:
            raise ValueError("Risk aversion must be positive.")
        
        self.data = data
        self.lambda_ = lambda_
    
    @abstractmethod
    def construct(self)-> jnp:
        """
        Abstract method to construct the portfolio.
        Subclasses must implement this method.

        Returns:
            np.ndarray: Portfolio weights.
        """
        pass

