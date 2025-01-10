from utils.data_loader import create_data

world_indices = [
    '^MERV',      # Argentina
    '^AXJO',      # Australia
    '^BVSP',      # Brazil
    '^GSPTSE',    # Canada
    '000001.SS',  # China (Shanghai Composite)
    '^STOXX50E',  # Euro Area (Euro Stoxx 50)
    '^FCHI',      # France
    '^GDAXI',     # Germany
    '^HSI',       # Hong Kong
    '^NSEI',      # India (NIFTY 50)
    '^BSESN',     # India (BSE SENSEX)
    '^JKSE',      # Indonesia
    'FTSEMIB.MI', # Italy
    '^N225',      # Japan
    '^MXX',       # Mexico
    '^STI',       # Singapore
    '^KS11',      # South Korea
    '^IBEX',      # Spain
    '^SSMI',      # Switzerland
    'XU100.IS',   # Turkey
    '^FTSE',      # United Kingdom
    '^GSPC',      # United States (S&P 500)
    '^DJI',       # United States (Dow Jones)
    '^IXIC',      # United States (Nasdaq Composite)
    '^BFX',       # Belgium
    '^AEX',       # Netherlands
    '^ATX',        # Austria
    '^NYA',
    '^XAX',
    '^BUK100P',
    '^RUT',
    '^VIX',
    '^TWII',
]

start_date = '2015-1-2'
end_date = '2024-1-1'
inteval = '1d'
path = '/Users/nicolasnguyen/Documents/Projets/PortfolioVB/saved/daily_data'

data_df = create_data(
    world_indices,
    start_date,
    end_date,
    inteval,
    path
)