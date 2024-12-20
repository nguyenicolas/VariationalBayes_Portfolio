from portfolio.base import PortfolioConstructor

class Markowitz(PortfolioConstructor):
    """
    Mean-Variance/Markowitz Portfolio Constructor
    """
    def __init__(self, data, lambda_) -> None:
        super().__init__(data, lambda_)

    def construct(self) -> jnp:
        #returns =
        # cov_matrix =  
        pass