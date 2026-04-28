from forecasting.prophet_model import ProphetVolatilityModel
from rl.rebalance_agent import PortfolioAgent
import pandas as pd

df = pd.read_csv("data/sample_data.csv")

model = ProphetVolatilityModel()
model.fit(df)

pred = model.predict()

yhat = pred['yhat'].values

returns = (yhat[1:] - yhat[:-1])/ yhat[:-1]

state = returns[-2:]

# RL Agent

agent = PortfolioAgent(n_assets=2)

weights = agent.act(state)

print("Predictions:\n", pred.head())
print("State (returns used): ", state)
print("Weights: ", weights)