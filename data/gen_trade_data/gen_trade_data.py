import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
from sklearn.linear_model import *
from joblib import Parallel, delayed
from itertools import islice
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import datetime as dt
import requests
import sys
import re
import itertools
import os
import talib
import logging
from decimal import Decimal
import io
import sys
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import warnings

warnings.filterwarnings('ignore')
params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0
param_dict = dict(params_df.iloc[param_row, :])
dataset_name = param_dict["dataset_name"]


# dataset_name = "AUD_CAD_H1"
# dataset_name = "EUR_USD_H1"
root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data" 
saved_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data/saved_data" 
dataset_path = f"{root_data_dir}/{dataset_name}_processed_data.csv"


# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
# config_path = "/Users/newuser/Projects/robust-algo-trader/settings/config.json"
with open(config_path) as f:
  config = json.load(f)  
# Get the take_profit and stop_loss levels from the config file
config_settings = config["trading_settings"][dataset_name]
window_size = config["window_size"]
tp = config_settings["take_profit"]
sl = config_settings["stop_loss"]
device = config["device"]
model_path = config['paths']["model_39_dir"]

df = pd.read_csv(dataset_path)
df = df.rename(columns={'time': 'Time'})
# filter only up to 2015
df = df[df['Time'] < '2015-01-01 00:00:00']
df['Index'] = df.index
y = df[['Close']]
offset = y.index[0]

# Set a random seed for reproducibility
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
# Define the model
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 5) 
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(86528, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply the convolutional layers with pooling and dropout
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
# Load the trained model
cnn = CNNet()
cnn.load_state_dict(torch.load(model_path))
cnn.to(device)
cnn.eval()



def save_setup_graph(subset_df, position, label, index):
    green_df = subset_df[subset_df['Close'] > subset_df['Open']].copy()
    green_df["Height"] = green_df["Close"] - green_df["Open"]
    red_df = subset_df[subset_df['Close'] < subset_df['Open']].copy()
    red_df["Height"] = red_df["Open"] - red_df["Close"]
    
    # if green_df or red_df is empty, then return
    # if the length of green_df and red_df is less than the window_size, then return
    if green_df.empty and red_df.empty:
        return
    
    # switch to "Agg" backend to prevent showing
    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(8, 3))
    
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
    
    buf = io.BytesIO()
    plt.savefig(buf, dpi=128, bbox_inches="tight", format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    # Get the model output
    output = cnn(image)
    output_item = output.item()
    buf.close()
    
    pred_label = 1 if output_item > 0.5 else 0
    if pred_label == label:
        # name should be the index of the first row in the subset_df
        # Check if the directory exists, if not, create it
        directory_path = f"{saved_data_dir}/{dataset_name}/{label}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(f"{directory_path}/{dataset_name}_{index}.png", dpi=128, bbox_inches="tight")
    # close the figure
    plt.close()
    

trades = []

# loop through all rows in the dataframe
for index in range(window_size, len(df)):
    i = index + offset

    if ((df.loc[i, "MACD_Crossover_Change"] > 0) and
       (df.loc[i, "Close"] > df.loc[i, "SMA_20"]) and 
       (df.loc[i, "Close"] > df.loc[i, "SMA_30"])):
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
            "Time": df.loc[i, "Time"],
            "label": None,
        }
        # add a second loop to check if the current close price is greater than the take profit price
        # or less than the stop loss price
        for k in range(index+1, len(df)):
            j = k + offset
            if df.loc[j, "Close"] >= tp_price:
                local_order["label"] = 1
                local_order["close_time"] = df.loc[j, "Time"]
                break
            elif df.loc[j, "Close"] <= sl_price:
                local_order["label"] = 0
                local_order["close_time"] = df.loc[j, "Time"]
                break
        
        if local_order["label"] is None:
            break    
        # create set-up graph for local_order
        # subset_df should be a df with window_size rows from i-window_size to i
        subset_df = df.loc[i-window_size:i]
        save_setup_graph(subset_df, current_position, local_order["label"], i)
        trades.append(local_order)
    elif ((df.loc[i, "MACD_Crossover_Change"] < 0) and 
          (df.loc[i, "Close"] < df.loc[i, "SMA_20"]) and 
         (df.loc[i, "Close"] < df.loc[i, "SMA_30"])):   
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
            "Time": df.loc[i, "Time"],
            "label": None,
        }
        
        for k in range(index+1, len(df)):
            j = k + offset
            if df.loc[j, "Close"] <= tp_price:
                local_order["label"] = 1
                local_order["close_time"] = df.loc[j, "Time"]
                break
            elif df.loc[j, "Close"] >= sl_price:
                local_order["label"] = 0
                local_order["close_time"] = df.loc[j, "Time"]
                break
        
        if local_order["label"] is None:
            break
        subset_df = df.loc[i-window_size:i]
        save_setup_graph(subset_df, current_position, local_order["label"], i)
        trades.append(local_order)
        
# trades_df = pd.DataFrame(trades)
print("Done!")
