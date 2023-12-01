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

y = df[['Close']]
offset = y.index[0]

def save_setup_graph(subset_df, position, label):
    plt.figure(figsize=(8,6)) # Increase the figure size to fit two subplots
    plt.subplot(2, 1, 1)
    # plt.plot(subset_df["Close"], label="Close")
    plt.plot(subset_df["SMA_20"], label="SMA_20")
    # plt.plot(df["SMA_30"], label="SMA_30")
    plt.plot(subset_df["SMA_50"], label="SMA_50")
    plt.plot(subset_df["SMA_100"], label="SMA_100")
    close_price = subset_df["Close"].iloc[-1]
    if position == 1:
        plt.axhspan(close_price, close_price + tp, facecolor="green", xmin= 0.96, alpha=0.5) 
        plt.axhspan(close_price - sl, close_price, facecolor="red", xmin= 0.96, alpha=0.5)
    else:
        plt.axhspan(close_price, close_price + sl, facecolor="red", xmin= 0.96, alpha=0.5) 
        plt.axhspan(close_price - tp, close_price, facecolor="green", xmin= 0.96, alpha=0.5)
    plt.xticks([])
    plt.yticks([])
    
    # Plot the MACD graph below the price chart
    plt.subplot(2, 1, 2) # Create a subplot for the MACD graph
    plt.plot(subset_df["MACD"], label="MACD") # Add this line to plot the MACD column
    plt.plot(subset_df["MACD_Signal"], label="MACD_Signal") # Add this line to plot the MACD_Signal column
    plt.bar(subset_df.index, subset_df["MACD_Hist"], label="MACD_Hist") # Add this line to plot the MACD_Hist column as a bar chart
    plt.legend()
    plt.title("MACD chart")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    
    save_path = f"/projects/genomic-ml/da2343/ml_project_2/data/EURUSD/{label}"
    # name should be the index of the first row in the subset_df
    plt.savefig(f"{save_path}/{subset_df.index[0]}.png", dpi=64, bbox_inches="tight")
    # close the figure
    plt.close()
    


trades = []
window_size = 24 * 10

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
        
        # create set-up graph for local_order
        # subset_df should be a df with window_size rows from i-window_size to i
        subset_df = df.loc[i-window_size:i]
        save_setup_graph(subset_df, current_position, local_order["label"])
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
        save_setup_graph(subset_df, current_position, local_order["label"])
        trades.append(local_order)
        
# trades_df = pd.DataFrame(trades)
print("Done!")
