import jax.numpy as jnp
import portfolio.VB as VB

path_file = '/Users/nicolasnguyen/Documents/Projets/PortfolioVB/saved/daily_data.npy'
data_df = jnp.load(path_file)
data_train = jnp.array(data_df)[200:1000, :80]

params = {
    'lambda_' : 1,
    'nb_inner_iter' : 5,
    'nb_GD_iter' : 100,
    'step_size_GD' : 1e2,
}

VB_GW_Ptf = VB.VB_Portfolio(data_train, params)
deltaVB = VB_GW_Ptf.construct()

print(jnp.round(deltaVB, 3))