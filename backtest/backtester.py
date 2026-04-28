import numpy as np

def backtest(returns, weights):
    # returns - matrix of asset returns over time
    # shape - (timestamp, n_assets) 
    # weights - portfolio allocation for each asset
    # shape - (n_assets,)

    portfolio = np.sum(returns*weights, axis = 1) # Portfolio return per day

    cummulative = np.cumsum(portfolio)

    # Cummulative potfolio performance
    return cummulative