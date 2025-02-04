import jax
from typing import Tuple
import jax.numpy as jnp
from portfolio.base import BayesianPortfolioConstructor

class VB_Portfolio(BayesianPortfolioConstructor):
    """Variational Bayes portfolio construction on Gaussian-Wishart model.
    """

    def __init__(self, data: jnp.array, params: dict) -> None:
        """Initialize Variational Bayes portfolio constructor.

        Args:
            data: jnp.array of size (n, d) where n is the sample size and d the number of assets.
            params: dict containing parameters.
        """
        super().__init__(data, params)
    
    def create_priors(self) -> None:
        return super().create_priors()
    
    def construct(self) -> jnp.array:
        """Returns decision vector with Gradient descent.

        Returns: 
            delta (jnp.array of size d): decision vector.
        """
        for _ in range(self.nb_GD_iter) :
            self.one_step_GD()
        return self.delta
    
    def init_phi(self) -> dict:
        """Initializes dictionary and store variational parameters.

        Returns: 
            phi (dict): dictionary containing variational parameters for the Gaussian-Wishart model.
        """
        phi = dict()
        phi['xi_y'], phi['Lambda_y'], phi['xi_mu'], phi['Lambda_mu'], phi['psi_Lambda'] = jnp.ones(self.d), jnp.eye(self.d), jnp.ones(self.d), jnp.eye(self.d), jnp.eye(self.d)
        return phi 
    
    def unpack_phi(self, phi: dict) -> Tuple:
        """Transform a dictionary of parameters into a tuple of these parameters.

        Args:
            phi (dict): dictionary containing variational parameters.
        Returns:
            tuple of variational parameters.
        """
        return phi['xi_y'], phi['Lambda_y'], phi['xi_mu'], phi['Lambda_mu'], phi['psi_Lambda']

    def fixed_point_operator(self, delta: jnp.array, phi: dict):
        """Fixed-point operator for the Gaussian-Wishart model.

        Args:
            delta (jnp.array): evaluation input of the function.
            phi (dict): dictionary of model parameters.
        
        Returns:
            phi (dict): dictionary of updated model parameters.
        """
        xi_y, Lambda_y, xi_mu, Lambda_mu, psi_Lambda = self.unpack_phi(phi)

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
        phi['xi_y'], phi['Lambda_y'], phi['xi_mu'], phi['Lambda_mu'], phi['psi_Lambda'] = new_xi_y, new_Lambda_y, new_xi_mu, new_Lambda_mu, new_psi_Lambda
        return phi
    
    def compute_objective(self, delta: jnp.array, phi: dict):
        """Objective function evaluated in delta with variational parameters phi.

        Args:
            delta (jnp.array): evaluation point of the objective function.
            phi (dict): dictionary of variational parameters.
        
        Returns: 
            float: objective function evaluated in delta.
        """

        xi_y, Lambda_y, xi_mu, Lambda_mu, psi_Lambda = self.unpack_phi(phi)
        nu_Lambda = self.n + self.nu_0 + 1

        self.mu_estimate = jnp.copy(xi_mu)
        self.Sigma_estimate = jnp.copy(jnp.linalg.inv(nu_Lambda * psi_Lambda))

        Lambda_mu_inv = jnp.linalg.inv(Lambda_mu)
        signLambda_, abslogdetLambda = jnp.linalg.slogdet(psi_Lambda)
        signy_, abslogdety = jnp.linalg.slogdet(Lambda_y)
        signmu_, abslogdetmu = jnp.linalg.slogdet(Lambda_mu)

        return (
            - 0.5 * nu_Lambda * jnp.trace(
                jnp.dot(
                    jnp.einsum('ij,ik->jk', self.data, self.data) - 2 * jnp.outer(jnp.sum(self.data, axis=0) + xi_y, xi_mu) + (self.n + 1)*(Lambda_mu_inv + jnp.outer(xi_mu, xi_mu)) + jnp.linalg.inv(Lambda_y) + jnp.outer(xi_y, xi_y) + self.psi_0_inv,
                    psi_Lambda
                    )
                )
            - 0.5 * jnp.trace(jnp.dot(Lambda_mu_inv + jnp.outer(xi_mu, xi_mu), self.Lambda_0))
            + jnp.dot(xi_mu.T, jnp.dot(self.Lambda_0, self.mu_0))
            + 0.5 * (self.n + self.nu_0 + 1) * signLambda_ * abslogdetLambda
            - 0.5 *(signy_ * abslogdety + signmu_ * abslogdetmu)
            - self.lambda_ * jnp.dot(delta.T, xi_y)
            )
    
    def fixed_point_solver(self, delta: jnp.array) -> dict:
        """Solve the fixed-point equation using a fixed number of iterations.
        
        Args: 
            delta (jnp.array): evaluation point.
        Returns:
            phi (dict): dictionary of variational parameters .
        """
        
        phi = self.init_phi()
        for _ in range(self.nb_inner_iter):
            phi = self.fixed_point_operator(delta, phi)
        return phi
    
    def objective(self, delta: jnp.array):
        """Compute objective function evaluated in a given decision.

        Args:
            delta (jnp.array): evaluation point.
        Return: 
            objective function evaluated in delta.
        """

        phi = self.fixed_point_solver(delta)
        return self.compute_objective(delta, phi)

    def compute_grad(self, delta: jnp.array):
        """Compute the gradient of the objective with respect to delta.
        
        Args:
            delta (jnp.array): evaluation point.
        Returns:
            gradient evaluated in delta.
        """
        return jax.grad(self.objective)(delta)
    
    
    def one_step_GD(self) -> None:
        """Performs one step of gradient descent.
        """
        step_size = self.step_size_GD
        delta_old = jnp.copy(self.delta)

        grad_evaluated = self.compute_grad(delta_old)
        norm_grad = jnp.linalg.norm(grad_evaluated)

        print(norm_grad)

        #self.VB_evaluate["gradient norm"]+=[norm_grad]
        delta_new = delta_old - step_size * grad_evaluated
        self.delta = jnp.copy(delta_new)