import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

path_file = '/Users/nicolasnguyen/Documents/Projets/PortfolioVB/saved/daily_data.npy'
data_df = np.load(path_file)
data = jnp.array(data_df)