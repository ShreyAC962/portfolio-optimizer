from fastapi import FastAPI
import numpy as np
from rl.rebalance_agent import PortfolioAgent

app = FastAPI()
agent = PortfolioAgent(2)

@app.get("/predict")
def predict():
    state = np.array([0.2,0.3])
    weights = agent.act(state)
    return {"allocation " : weights.tolist()}        
