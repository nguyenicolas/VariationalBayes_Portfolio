import numpy as np
import jax.numpy as jnp
from portfolio.base import PortfolioConstructor

class VB_Portfolio(PortfolioConstructor):
    """
    Variational Bayes portfolio construction general class
    """
    def __init__(self, data: jnp.array, lambda_: float, priors:dict) -> None:
        """
        Initialize Variational Bayes portfolio constructor.

        Args:
            priors: dictionary of prior parameters
        """
        super().__init__(data, lambda_)
        self.priors = priors
        self.delta0 = jnp.zeros(self.n)
    
    def construct(self) -> jnp.array:
        # perform gradient descent
        pass


    def fixed_point(self, delta, rho) -> dict:
        """
        
        """ 

        nu_Lambda = self.n + self.nu_0 + 1
        new_xi_y = xi_mu - (self.lambda_/nu_Lambda) * jnp.dot(jnp.linalg.inv(psi_Lambda), delta)
        new_Lambda_y = jnp.dot(nu_Lambda, psi_Lambda)
        new_xi_mu = (1/(self.n + 1)) * jnp.dot(
            jnp.linalg.inv(nu_Lambda * psi_Lambda + (self.Lambda_0/(self.n + 1)) ),
            jnp.dot(self.Lambda_0, self.mu_0) + nu_Lambda * jnp.dot(psi_Lambda, new_xi_y + jnp.sum(self.data, axis=0))
            )
        new_Lambda_mu =  (self.n + 1) * nu_Lambda * psi_Lambda + self.Lambda_0
        new_psi_Lambda = jnp.linalg.inv(
            jnp.linalg.inv(new_Lambda_y) + jnp.outer(new_xi_y, new_xi_y)
            + (self.n + 1) * (
                jnp.linalg.inv(new_Lambda_mu)
                + jnp.outer(new_xi_mu, new_xi_mu)
                )
            + jnp.einsum('ij,ik->jk', self.data, self.data)
            - 2 * jnp.outer(new_xi_y + jnp.sum(self.data, axis=0), new_xi_mu)
            + self.psi_0_inv
            )
        return new_xi_y, new_Lambda_y, new_xi_mu, new_Lambda_mu, nu_Lambda, new_psi_Lambda
        

    def objective_function(self, delta):
        pass

    def one_step_GD(self):
        pass

