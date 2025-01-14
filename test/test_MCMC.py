import jax.numpy as jnp
import portfolio.MCMC as MCMC

path_file = '/Users/nicolasnguyen/Documents/Projets/PortfolioVB/saved/daily_data.npy'
data_df = jnp.load(path_file)
data_train = jnp.array(data_df)[200:1000, :10]

params = {
    'lambda_' : 1,
    'M' : 5000,
    'nb_GD_iter' : 30,
    'step_size_GD' : 1e3,
}

MCMC_GW_Ptf = MCMC.MCMC_Portfolio(data_train, params)
deltaMCMC = MCMC_GW_Ptf.construct()

print(jnp.round(deltaMCMC, 3))