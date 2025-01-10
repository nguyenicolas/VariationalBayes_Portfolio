import jax.numpy as jnp

class Metrics:
    def __init__(self, delta: jnp.array, test_data: jnp.array) -> None:
        self.delta = delta
        self.test_data = test_data
    
    def SR(self):
        pass
    def CumWealth(self):
        pass