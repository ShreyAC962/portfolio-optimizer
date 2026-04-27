from prophet import Prophet # Used for time series forecasting
import pandas as pd

class ProphetVolatilityModel:
    def fit(self, df):
        # Prophet expects specific column names:
        # 'ds' = date/time column
        # 'y' = value we want to predict (here: price or volatility)
        df = df.rename(columns={
            "datae":"ds",
            "price" : "y"
        })
        self.model = Prophet()
        # Training the model with historical data
    
        self.model.fit(df) # Model learns patters like trends and seasonality


    # Makes future predictions
    def predict(self, future_days = 5):
        # Create a future dataframe with extra dates(includes past + future (5 days here))
        future = self.model.make_future_dataframe(periods=future_days)

        # Generate forecast for both past and future dates
        forecast = self.model.predict(future)


        # Return only important columns:
        # 'ds' = date
        # 'yhat' = predicted value
        return forecast[['ds', 'yhat']]