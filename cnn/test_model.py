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
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image
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
import torchvision
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
threshold = param_dict["threshold"]


# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
  config = json.load(f)
  
# Get the take_profit and stop_loss levels from the config file
config_settings = config["trading_settings"][dataset_name]
tp = config_settings["take_profit"]
sl = config_settings["stop_loss"]
window_size = config["window_size"]

# dataset_path = config_settings["dataset_path"]

root_data_dir = config["paths"]["oanda_dir"]
device = config["device"]
ml_model_path = config["paths"]["model_80_dir"]


# Set a random seed for reproducibility
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
transform = transforms.Compose( 
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), 
                          (0.5, 0.5, 0.5))]) 
# Define the model
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
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
cnn.load_state_dict(torch.load(ml_model_path))
cnn.to(device)
cnn.eval()


df = pd.read_csv(f"{root_data_dir}/{dataset_name}_processed_data.csv")
df = df.rename(columns={'time': 'Time'})
df['Index'] = df.index
y = df[['Close']]
offset = y.index[0]

trades = []

def save_setup_graph(subset_df, position, index):
    green_df = subset_df[subset_df['Close'] > subset_df['Open']].copy()
    green_df["Height"] = green_df["Close"] - green_df["Open"]
    red_df = subset_df[subset_df['Close'] < subset_df['Open']].copy()
    red_df["Height"] = red_df["Open"] - red_df["Close"]
    
    plt.switch_backend("Agg")
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
    # close the figure
    plt.close()
    return output_item
    

def create_trade_order(row, position, tp, sl):
    ask_price = row["Close"]
    tp_price = ask_price + tp if position == 1 else ask_price - tp
    sl_price = ask_price - sl if position == 1 else ask_price + sl

    trade_order = {
        "index": row.name,
        "ask_price": ask_price,
        "take_profit_price": tp_price,
        "stop_loss_price": sl_price,
        "position": position,
        # f"SMA_{timeperiod}": row[f"SMA_{timeperiod}"],
        "MACD": row["MACD"],
        "MACD_Signal": row["MACD_Signal"],
        "MACD_Hist": row["MACD_Hist"],
        "MACD_Crossover_Change" : row["MACD_Crossover_Change"],
        "RSI": row["RSI"],
        "ATR": row["ATR"],
        "ADX": row["ADX"],
        "AROON_Oscillator": row["AROON_Oscillator"],
        "WILLR": row["WILLR"],
        "OBV": row["OBV"],
        "CCI": row["CCI"],
        "PSAR": row["PSAR"],
        "AD": row["AD"],
        "ADOSC": row["ADOSC"],
        # "VOLUME_RSI": row["VOLUME_RSI"],
        # "MFI": row["MFI"],
        "Time": row["Time"],
        "close_time": None,
        "label": None,
    }
    return trade_order

try:
    # loop through all rows in the dataframe
    for index, row in df.iloc[window_size:].iterrows():
        # check if there are any open trades
        if len(trades) != 0:
            prev_trade = trades[-1]
            # check if the previous trade was a long trade
            if prev_trade["position"] == 1:
                if row["Close"] >= prev_trade["take_profit_price"] and prev_trade["label"] == None:
                    prev_trade["label"] = 1
                    prev_trade["close_time"] = row["Time"]
                    continue
                elif row["Close"] <= prev_trade["stop_loss_price"] and prev_trade["label"] == None:
                    prev_trade["label"] = 0
                    prev_trade["close_time"] = row["Time"]
                    continue
            else:
                if row["Close"] <= prev_trade["take_profit_price"] and prev_trade["label"] == None:
                    prev_trade["label"] = 1
                    prev_trade["close_time"] = row["Time"]
                    continue
                elif row["Close"] >= prev_trade["stop_loss_price"] and prev_trade["label"] == None:
                    prev_trade["label"] = 0
                    prev_trade["close_time"] = row["Time"]
                    continue
                    
            if prev_trade["label"] == None:
                continue
        
        # if there are no open trades, check if there is a crossover
        macd_crossover_change = row["MACD_Crossover_Change"]
        if macd_crossover_change > 0 or macd_crossover_change < 0:
            if ((row["MACD_Crossover_Change"] > 0) and
                (row["Close"] > row["SMA_20"]) and 
                (row["Close"] > row["SMA_30"])):
                current_position = 1 # long
            elif ((row["MACD_Crossover_Change"] < 0) and
                  (row["Close"] < row["SMA_20"]) and 
                  (row["Close"] < row["SMA_30"])):
                current_position = 0 # short
            else:
                continue
            
            # TODO: Dummy
            # local_order = create_trade_order(row, current_position, tp, sl)
            # trades.append(local_order) 

            # TODO: ML
            subset_df = df.loc[(index-window_size+1):(index)]
            output_item = save_setup_graph(subset_df, current_position, index)
            pred = 1 if output_item > threshold else 0
            # use that to execute a trade order
            if pred == 1:
                print("output_item: ", output_item)
                local_order = create_trade_order(row, current_position, tp, sl)
                trades.append(local_order) 
except Exception as e:
    print(e)
    
trades_df = pd.DataFrame(trades)
trades_df['Time'] = pd.to_datetime(trades_df['Time'])
trades_df['Year'] = trades_df['Time'].dt.year
trades_df['Return'] = np.where(trades_df['label'] == 1, 2, -1)

# Create Max Drawdown column
max_drawdown_df = trades_df.copy() 
max_drawdown_df = max_drawdown_df[['Year', 'Return', 'label']]
max_drawdown = 0
max_drawdown_column = []
for index, row in max_drawdown_df.iterrows():
    if row['Return'] == 2:
        max_drawdown = 0
    elif row['Return'] == -1:
        max_drawdown += 1
    max_drawdown_column.append(max_drawdown)
max_drawdown_df['Max Drawdown'] = max_drawdown_column
max_drawdown_df = max_drawdown_df.groupby(['Year']).agg({'Max Drawdown': 'max'}).reset_index()

# for each year, sum the returns and count the number of labels as trades
trades_df = trades_df.groupby(['Year']).agg({'Return': 'sum', 'label': 'count'}).reset_index()
trades_df['trades'] = trades_df['label']
trades_df.drop(['label'], axis=1, inplace=True)
trades_df = pd.merge(trades_df, max_drawdown_df, on='Year')
trades_df['Percent Return'] = (trades_df['Return'] / trades_df['trades']) * 100
trades_df['dataset_name'] = dataset_name
trades_df['threshold'] = threshold

# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
trades_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!")