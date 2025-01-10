from portfolio.base import PortfolioConstructor
import jax.numpy as jnp

class EqualWeight(PortfolioConstructor):
    def __init__(self, data: jnp.array, params: dict) -> None:
        super().__init__(data, params)
    def construct(self) -> jnp.array:
        return jnp.ones(self.d) / self.d