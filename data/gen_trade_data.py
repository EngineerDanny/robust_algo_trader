import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
from sklearn.linear_model import *
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sktime.forecasting.model_selection import SlidingWindowSplitter
from joblib import Parallel, delayed
from itertools import islice
import json
import warnings
import matplotlib.pyplot as plt

root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data" 
dataset_path = f"{root_data_dir}/EURUSD/EURUSD_H1_200702210000_202304242100_Update.csv"
# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
  config = json.load(f)
  
dataset_name = "EURUSD_H1"
# Get the take_profit and stop_loss levels from the config file
tp = config["trading_settings"][dataset_name]["take_profit"]
sl = config["trading_settings"][dataset_name]["stop_loss"]
df = pd.read_csv(dataset_path, index_col=0)
df['Index'] = df.index

y = df[['Close']]
offset = y.index[0]

def save_setup_graph(subset_df, position, label, index):
    green_df = subset_df[subset_df['Close'] > subset_df['Open']].copy()
    green_df["Height"] = green_df["Close"] - green_df["Open"]
    red_df = subset_df[subset_df['Close'] < subset_df['Open']].copy()
    red_df["Height"] = red_df["Open"] - red_df["Close"]
    
    fig = plt.figure(figsize=(8,3))
    
    ##Grey Lines
    plt.vlines(x=green_df["Index"], 
            ymin=green_df["Low"], 
            ymax=green_df["High"],
            color="green")
    plt.vlines(x=red_df["Index"], 
            ymin=red_df["Low"], 
            ymax=red_df["High"],
            color="orangered")
    ##Green Candles
    plt.bar(x=green_df["Index"], 
            height=green_df["Height"], 
            bottom=green_df["Open"], 
            color="green")
    ##Red Candles
    plt.bar(x=red_df["Index"], 
            height=red_df["Height"], 
            bottom=red_df["Close"], 
            color="orangered")
    
    plt.plot(subset_df["SMA_20"], label="SMA_20")
    plt.plot(subset_df["SMA_30"], label="SMA_30")
    
    close_price = subset_df["Close"].iloc[-1]
    
    tp_eps = tp + 0.0025
    sl_eps = sl + 0.0025
    
    sl_eps = sl
    tp_eps = tp
    
    if position == 1:
        plt.axhspan(close_price, close_price + tp_eps, facecolor="green", xmin= 0.96, alpha=0.9) 
        plt.axhspan(close_price - sl_eps, close_price, facecolor="orangered", xmin= 0.96, alpha=0.9)
    else:
        plt.axhspan(close_price, close_price + sl_eps, facecolor="orangered", xmin= 0.96, alpha=0.9) 
        plt.axhspan(close_price - tp_eps, close_price, facecolor="green", xmin= 0.96, alpha=0.9)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    
    save_path = f"/projects/genomic-ml/da2343/ml_project_2/data/EURUSD/{label}"
    # name should be the index of the first row in the subset_df
    plt.savefig(f"{save_path}/{index}.png", dpi=128, bbox_inches="tight")
    # close the figure
    plt.close()
    


trades = []
window_size = 24 * 5

# loop through all rows in the dataframe
for index in range(window_size, len(df)):
    i = index + offset

    if df.loc[i, "MACD_Crossover_Change"] > 0:
        ask_price = df.loc[i, "Close"]
        tp_price = ask_price + tp
        sl_price = ask_price - sl
        current_position = 1

        local_order = {
            "index": i,
            "ask_price": ask_price,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "position": current_position,
            # f"SMA_{timeperiod}": df.loc[i, f"SMA_{timeperiod}"],
            "MACD": df.loc[i, "MACD"],
            "MACD_Signal": df.loc[i, "MACD_Signal"],
            "MACD_Hist": df.loc[i, "MACD_Hist"],
            "RSI": df.loc[i, "RSI"],
            "ATR": df.loc[i, "ATR"],
            "ADX": df.loc[i, "ADX"],
            "AROON_Oscillator": df.loc[i, "AROON_Oscillator"],
            "WILLR": df.loc[i, "WILLR"],
            "OBV": df.loc[i, "OBV"],
            "CCI": df.loc[i, "CCI"],
            "PSAR": df.loc[i, "PSAR"],
            "AD": df.loc[i, "AD"],
            "ADOSC": df.loc[i, "ADOSC"],
            "VOLUME_RSI": df.loc[i, "VOLUME_RSI"],
            "MFI": df.loc[i, "MFI"],
            "Date_Time": df.loc[i, "Date_Time"],
            "label": None,
        }
        # add a second loop to check if the current close price is greater than the take profit price
        # or less than the stop loss price
        for k in range(index+1, len(df)):
            j = k + offset
            if df.loc[j, "Close"] >= tp_price:
                local_order["label"] = 1
                local_order["close_time"] = df.loc[j, "Date_Time"]
                break
            elif df.loc[j, "Close"] <= sl_price:
                local_order["label"] = 0
                local_order["close_time"] = df.loc[j, "Date_Time"]
                break
        
        if local_order["label"] is None:
            break    
        
        
        # create set-up graph for local_order
        # subset_df should be a df with window_size rows from i-window_size to i
        subset_df = df.loc[i-window_size:i]
        save_setup_graph(subset_df, current_position, local_order["label"], i)
        trades.append(local_order)
        
        
    elif df.loc[i, "MACD_Crossover_Change"] < 0:   
        ask_price = df.loc[i, "Close"]  
        tp_price = ask_price - tp
        sl_price = ask_price + sl
        current_position = 0

        local_order = {
            "index": i,
            "ask_price": ask_price,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "position": current_position,
            # f"SMA_{timeperiod}": df.loc[i, f"SMA_{timeperiod}"],
            "MACD": df.loc[i, "MACD"],
            "MACD_Signal": df.loc[i, "MACD_Signal"],
            "MACD_Hist": df.loc[i, "MACD_Hist"],
            "RSI": df.loc[i, "RSI"],
            "ATR": df.loc[i, "ATR"],
            "ADX": df.loc[i, "ADX"],
            "AROON_Oscillator": df.loc[i, "AROON_Oscillator"],
            "WILLR": df.loc[i, "WILLR"],
            "OBV": df.loc[i, "OBV"],
            "CCI": df.loc[i, "CCI"],
            "PSAR": df.loc[i, "PSAR"],
            "AD": df.loc[i, "AD"],
            "ADOSC": df.loc[i, "ADOSC"],
            "VOLUME_RSI": df.loc[i, "VOLUME_RSI"],
            "MFI": df.loc[i, "MFI"],
            "Date_Time": df.loc[i, "Date_Time"],
            "label": None,
        }
        
        for k in range(index+1, len(df)):
            j = k + offset
            if df.loc[j, "Close"] <= tp_price:
                local_order["label"] = 1
                local_order["close_time"] = df.loc[j, "Date_Time"]
                break
            elif df.loc[j, "Close"] >= sl_price:
                local_order["label"] = 0
                local_order["close_time"] = df.loc[j, "Date_Time"]
                break
        
        if local_order["label"] is None:
            break
        
        subset_df = df.loc[i-window_size:i]
        save_setup_graph(subset_df, current_position, local_order["label"], i)
        trades.append(local_order)
        
# trades_df = pd.DataFrame(trades)
print("Done!")
