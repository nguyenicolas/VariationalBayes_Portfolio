from portfolio.base import PortfolioConstructor
import jax.numpy as jnp

class Markowitz(PortfolioConstructor):
    """
    Mean-Variance/Markowitz Portfolio Constructor
    """
    def __init__(self, data, lambda_) -> None:
        super().__init__(data, lambda_)

    def construct(self) -> jnp.array
        #returns =
        # cov_matrix =  
        pass