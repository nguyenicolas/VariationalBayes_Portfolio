import numpy as np
import yfinance as yf
import pandas as pd

def create_data(indices:list, start_date: str, end_date:str, interval:str, path:str) -> None:
    """
    create and save a data of linear returns for portfolio construction.
    
    Args:
        indices (str): Path to the CSV file.
        start_date (str): start date of index tracking
        end_date (str): end date of index tracking
        interval (str): time interval of index tracking
        path (str): path to save
    """

    df = yf.download(indices, start=start_date, end=end_date, interval=interval)['Close']
    linear_returns = df.pct_change()
    linear_returns.fillna(0, inplace=True)
    linear_returns.to_numpy()
    np.save(path, linear_returns)