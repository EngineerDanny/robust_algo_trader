import pandas as pd
import numpy as np
import streamlit as st
import darts
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

import darts
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.model_selection import train_test_split
from darts.dataprocessing.transformers import MissingValuesFiller

# models
from darts.models import AutoARIMA, Theta, ExponentialSmoothing, Prophet 
from darts.models import XGBModel, LightGBMModel

# evaluation
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle

# settings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction APP")

stocks =("TSLA", "AAPL", "MSFT", "005930.KS")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading Data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name= 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name= 'stock_close'))
    fig.layout.update(title_text= "Time Series Data", xaxis_rangeslider_visible= True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df= data[['Date','Close']]

ts = TimeSeries.from_dataframe(df, time_col='Date', value_cols='Close', fill_missing_dates=True, freq='D')
#Treat Missing Values
filler = MissingValuesFiller()
ts= filler.transform(ts, method="quadratic")

# Train-test split
train_data, test_data = ts[:-612], ts[-612:]   #test size = len(ts)*0.2

encoders = {"datetime_attribute": {"past": ["month", "year"]}}
# Set  lags and output_chunk_length parameters
new_lags = 36
new_output_chunk_length = 365

# Train forecasting models
models = {
    # "Exponential Smoothing": ExponentialSmoothing(),
    # "AutoARIMA": AutoARIMA(),
    "Theta Model": Theta(),
    # "Prophet": Prophet(),
    "Xgboost": XGBModel(lags=new_lags, 
                        # lags_past_covariates = new_lags,
                               output_chunk_length=new_output_chunk_length, 
                               random_state=0,
                            #    add_encoders=encoders,                             
                               multi_models=False
                               ),
    
    # "LightGBM": LightGBMModel(lags=new_lags, lags_past_covariates = new_lags,
    #                            output_chunk_length=new_output_chunk_length, 
    #                            random_state=0,
    #                            add_encoders=encoders,                               
    #                            multi_models=False)
}

# Evaluate models and store the metrics
eval_metrics = {}
for model_name, model_instance in models.items():
    model_instance.fit(train_data)
    forecast = model_instance.predict(len(test_data))
    eval_metrics[model_name] = mape(test_data, forecast)


# Select model with the lowest metric value
best_model = min(eval_metrics, key=eval_metrics.get)
best_model_instance = models[best_model]

# Display evaluation metrics
st.subheader("Evaluation Metrics: MAPE")
st.write(eval_metrics)

st.subheader("Best Model")
st.write(best_model)

# Make predictions using the best model
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years*365
if st.button("Make Future Predictions"):
    # Retrain the best model on the entire dataset
    best_model_instance.fit(ts)
    
    # Generate future predictions
    future_predictions = best_model_instance.predict(period)

    st.subheader("Future Predictions")
    
    actual_df = ts.pd_dataframe()
    predicted_df = future_predictions.pd_dataframe()
    st.write(predicted_df.head(period))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df.index, y=actual_df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['Close'], name='Predicted', line=dict(color='red')))
    
    st.plotly_chart(fig)
     