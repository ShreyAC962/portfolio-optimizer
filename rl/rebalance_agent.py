import numpy as np

# Simple Portfolio Optimization Agent - decision-maker for asset allocation

class PortfolioAgent:
    def __init__(self, n_assets):
        '''
        # Initialize equal weights across all assets
        # Example: if 3 assets → [0.33, 0.33, 0.33]
        # This means we start with a balanced portfolio
        '''
        self.weights = np.ones(n_assets) /n_assets 

    def act(self, state):
        # state -> predicted returns or signals for each asset
        # convert raw predictions into positive values - exp(makes all values positive and emphasizes strong signals)
        weights = np.exp(state)

        # Valid Portfolio Allocation - Normalize so weights sum to 1(100% distribution of capital across assets)
        return weights/ np.sum(weights)
    
    def update(self, reward):
        # If reward positive - strengthen current strategy or else weaken it
        learning_rate = 0.1
        self.weights = self.weights + learning_rate*reward

        # Keep weight valid - no negative
        self.weights = np.maximum(self.weights, 0)

        # Normalize again so total = 1
        self.weights = self.weights/np.sum(self.weights)