import jax
import jax.numpy as jnp
from portfolio.base import PortfolioConstructor

class VB_Portfolio(PortfolioConstructor):
    """
    Variational Bayes portfolio construction general class
    """
    def __init__(self, data: jnp.array, params: dict) -> None:
        """
        Initialize Variational Bayes portfolio constructor.

        Args:

        """
        super().__init__(data, params)
        self.delta = jnp.zeros(self.d)
        self.create_priors()
    
    def create_priors(self):
        self.mu_0 = jnp.mean(self.data, axis=0)
        self.Lambda_0 = jnp.eye(self.d)
        self.nu_0 = self.n + self.d
        cov = jnp.cov(self.data.T)
        self.psi_0_inv = cov
        self.psi_0 = jnp.linalg.inv(cov) 
    
    #@jax.jit
    def construct(self) -> jnp.array:
        # perform gradient descent
        for _ in range(self.nb_GD_iter) :
            self.one_step_GD()
        #delta_normalized = self.delta #/ np.linalg.norm(self.delta, 1)
        #self.VB_evaluate["delta"] = delta_normalized
        return self.delta
    
    def unpack_phi(self, phi):
        return phi['xi_y'], phi['Lambda_y'], phi['xi_mu'], phi['Lambda_mu'], phi['psi_Lambda']

    def init_phi(self):
        phi = dict()
        phi['xi_y'], phi['Lambda_y'], phi['xi_mu'], phi['Lambda_mu'], phi['psi_Lambda'] = jnp.ones(self.d), jnp.eye(self.d), jnp.ones(self.d), jnp.eye(self.d), jnp.eye(self.d)
        return phi 
    
    def fixed_point_operator(self, delta, phi):
        """
        ...
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
    
    def compute_objective(self, phi, delta):

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
    
    #@jax.jit
    def fixed_point_solver(self, delta):
        """Solve the fixed-point equation φ = g(φ, δ) using a fixed number of iterations."""
        
        phi = self.init_phi()
        for _ in range(self.nb_inner_iter):
            phi = self.fixed_point_operator(delta, phi)
        return phi

    #@jax.jit
    def objective(self, delta):
        """
        Compute the objective as a function of δ
        """

        phi = self.fixed_point_solver(delta)
        return self.compute_objective(phi, delta)

    #@jax.jit
    def compute_grad(self, delta):
        """Compute the gradient of the objective with respect to δ."""
        return jax.grad(self.objective)(delta)
    
    
    def one_step_GD(self):
        step_size = self.step_size_GD
        delta_old = jnp.copy(self.delta)

        grad_evaluated = self.compute_grad(delta_old)
        norm_grad = jnp.linalg.norm(grad_evaluated)

        print(norm_grad)

        #self.VB_evaluate["gradient norm"]+=[norm_grad]
        delta_new = delta_old - step_size * grad_evaluated
        self.delta = jnp.copy(delta_new)