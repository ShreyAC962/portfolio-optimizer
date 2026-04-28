import numpy as np

# Function to measure risk-adjusted performance of a strategy
def sharpe_ratio(returns):
    
    # returns → list/array of portfolio returns over time
    # Example: [0.01, -0.02, 0.03, 0.015]
    
    # Mean return → average profit of strategy
    mean_return = np.mean(returns)
    
    # Standard deviation → risk (how much returns fluctuate)
    std_return = np.std(returns)
    
    # Sharpe ratio = reward per unit of risk
    # Higher value = better strategy (more return, less risk)
    return mean_return / std_return