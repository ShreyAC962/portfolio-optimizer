import torch
import torch.nn as nn # neural network model used to build models like LSTM, layers etc

class VolatilityNey(nn.Module):
    # This method initializes the model (defines layers)
    def __init__(self):
        # Call parent class - required for Pytorch models
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32,1)   # Takes 32 features from LSTM and outputs 1 value (prediction)

    def forward(self, x):
        # Pass input through LSTM
        # out → output for all time steps
        # _ → hidden states (we ignore them here)
        out, _ = self.lstm(x)

        # Take only the LAST time step output
        # out[:, -1, :] means:
        # all batches, last time step, all features
        last_output = out[:,-1,:]

        # Pass last output through fully connected layer
        # This gives final prediction (e.g., next volatility value)
        return self.fc(last_output)

