import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
from sklearn.linear_model import *
from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sktime.forecasting.model_selection import SlidingWindowSplitter
from joblib import Parallel, delayed
from itertools import islice
import json
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
forecast_horizon = param_dict["fh"]
window_size = param_dict["window_size"]
algorithm = param_dict["algorithm"]
train_size = param_dict["train_size"]
timeperiod = param_dict["sma"]

slope_threshold = param_dict["slope_threshold"]
year = str(param_dict["year"])
step_length = param_dict["step_length"]

root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data" 
dataset_dict = {
    "EURUSD_H1" : f"{root_data_dir}/EURUSD/EURUSD_H1_200702210000_202304242100_Update.csv",
    "USDJPY_H1" : f"{root_data_dir}/USDJPY/USDJPY_H1_200705290000_202307282300_Update.csv",
    "GBPUSD_H1" : f"{root_data_dir}/GBPUSD/GBPUSD_H1_200704170000_202307282300_Update.csv",
    "AUDUSD_H1" : f"{root_data_dir}/AUDUSD/AUDUSD_H1_200704170000_202307282300_Update.csv",
    "USDCAD_H1" : f"{root_data_dir}/USDCAD/USDCAD_H1_200705300000_202307282300_Update.csv",
    "USDCHF_H1" : f"{root_data_dir}/USDCHF/USDCHF_H1_200704170000_202307282300_Update.csv",
    # "NZDUSD_H1" : f"{root_data_dir}/NZDUSD/NZDUSD_H1_200704170000_202307282300_Update.csv",
    "EURJPY_H1" : f"{root_data_dir}/EURJPY/EURJPY_H1_200705300000_202307282300_Update.csv",
    "EURGBP_H1" : f"{root_data_dir}/EURGBP/EURGBP_H1_200703270000_202307282300_Update.csv",
}

dataset_path = dataset_dict[dataset_name]
# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
  config = json.load(f)
# Get the take_profit and stop_loss levels from the config file
tp = config["trading_settings"][dataset_name]["take_profit"]
sl = config["trading_settings"][dataset_name]["stop_loss"]

def extract_df(path, year, train_size):
    df = pd.read_csv(path, index_col=0)
    # Convert the Date_Time column to datetime format
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    # Filter the dataframe by the year condition
    prev_df = df[df['Date_Time'].dt.year < int(year)]
    prev_df = prev_df.tail(train_size)
    current_yr_df = df[df['Date_Time'].dt.year == int(year)]
    filtered_df = pd.concat([prev_df, current_yr_df])
    return filtered_df
    
df = extract_df(dataset_path, year, train_size)



learner_dict = {
    "LinearRegression": LinearRegression(),
}

splitter = SlidingWindowSplitter(window_length=train_size, 
                                 fh=np.arange(1, forecast_horizon + 1), 
                                 step_length=step_length)

accuracy_df_list = []
mapes = []
orders = []
outcomes = []

y = df[[f'SMA_{timeperiod}']]
offset = y.index[0]


def forecast_isolate(j):
    # get the train and test indices
    splitter_y = splitter.split(y)
    train_idx, test_idx = next(islice(splitter_y, j, None))

    train_idx = train_idx + offset
    test_idx = test_idx + offset
    
    # get the train and test data
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # create a forecaster object using LinearRegression and recursive strategy
    # regressor = learner_dict[algorithm]
    # forecaster = make_reduction(regressor, 
    #                             window_length=window_size, 
    #                             strategy="recursive")
    # forecaster.fit(y_train)
    # fh = ForecastingHorizon(y_test.index, is_relative=False)
    # y_pred = forecaster.predict(fh)
    # mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    
    # x = np.arange(1, len(y_pred) + 1)
    # slope, intercept = np.polyfit(x, y_pred, 1)
    # y_fit = np.polyval([slope, intercept], x)
    # mse = mean_squared_error(y_pred, y_fit)
    # mse_scaled = mse * 10e8
    # slope_scaled = slope * 10e5
    
    mape = 0
    current_position = None
    outcome = None
    local_order = {"index": None}
    has_traded = False
    
    for i in test_idx:
        # check if there is a buy MACD crossover
        # if df.loc[i, "MACD_Crossover_Change"] > 0 and \
        #     has_traded == False and \
        #     df.loc[i, "Close"] > df.loc[i, f"SMA_{timeperiod}"] and \
        #     df.loc[i, "MACD"] < 0:
        # if df.loc[i, "AROON_Oscillator"] <= -100 and \
        if df.loc[i, "MACD_Crossover_Change"] > 0 and \
            has_traded == False:
            ask_price = df.loc[i, "Close"]
            tp_price = ask_price + tp
            sl_price = ask_price - sl
            current_position = 1
            
            # tp_price = ask_price - tp
            # sl_price = ask_price + sl
            # current_position = 0
            
            local_order = {
                "index": i,
                "ask_price": ask_price,
                "take_profit_price": tp_price, 
                "stop_loss_price": sl_price, 
                "position": current_position,
                f"SMA_{timeperiod}" : df.loc[i, f"SMA_{timeperiod}"],
                "MACD" : df.loc[i, "MACD"],
                'MACD_Signal' : df.loc[i, "MACD_Signal"],
                "MACD_Hist" : df.loc[i, "MACD_Hist"],
                "RSI" : df.loc[i, "RSI"],
                "ATR" : df.loc[i, "ATR"],
                "ADX" : df.loc[i, "ADX"],
                "AROON_Oscillator" : df.loc[i, "AROON_Oscillator"],
                "WILLR" : df.loc[i, "WILLR"],

                "OBV" : df.loc[i, "OBV"],
                "CCI" : df.loc[i, "CCI"],
                "PSAR" : df.loc[i, "PSAR"],
                "AD" : df.loc[i, "AD"],
                "ADOSC" : df.loc[i, "ADOSC"],
                "VOLUME_RSI" : df.loc[i, "VOLUME_RSI"],                
                "MFI" : df.loc[i, "MFI"],
                "Date_Time" : df.loc[i, "Date_Time"],
                "label": None,
            }
            has_traded = True
            
        # elif df.loc[i, "MACD_Crossover_Change"] < 0 and \
        #     has_traded == False and \
        #     df.loc[i, "Close"] < df.loc[i, f"SMA_{timeperiod}"] and \
        #     df.loc[i, "MACD"] > 0:    
        # elif df.loc[i, "AROON_Oscillator"] >= 100 and \
        elif df.loc[i, "MACD_Crossover_Change"] < 0 and \
            has_traded == False:    
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
                f"SMA_{timeperiod}" : df.loc[i, f"SMA_{timeperiod}"],
                "MACD" : df.loc[i, "MACD"],
                'MACD_Signal' : df.loc[i, "MACD_Signal"],
                "MACD_Hist" : df.loc[i, "MACD_Hist"],
                "RSI" : df.loc[i, "RSI"],
                "ATR" : df.loc[i, "ATR"],
                "ADX" : df.loc[i, "ADX"],
                "AROON_Oscillator" : df.loc[i, "AROON_Oscillator"],
                "WILLR" : df.loc[i, "WILLR"],
                "OBV" : df.loc[i, "OBV"],
                "CCI" : df.loc[i, "CCI"],
                "PSAR" : df.loc[i, "PSAR"],
                "AD" : df.loc[i, "AD"],
                "ADOSC" : df.loc[i, "ADOSC"],
                "VOLUME_RSI" : df.loc[i, "VOLUME_RSI"],                
                "MFI" : df.loc[i, "MFI"],
                "Date_Time" : df.loc[i, "Date_Time"],
                "label": None,
            }
            has_traded = True
        
        if has_traded == True:
            close_price = df.loc[i, "Close"]
            if current_position == 1:
                if close_price >= tp_price:
                    outcome = 1
                    local_order["label"] = outcome
                    break
                elif close_price <= sl_price:
                    outcome = 0
                    local_order["label"] = outcome
                    break
            else:
                if close_price <= tp_price:
                    outcome = 1
                    local_order["label"] = outcome
                    break
                elif close_price >= sl_price:
                    outcome = 0
                    local_order["label"] = outcome
                    break
    
    # if abs(slope_scaled) > slope_threshold:
    #     has_traded = False
    #     if slope > 0:
    #         for i in test_idx:
    #             # check if there is a buy MACD crossover
    #             if df.loc[i, "MACD_Crossover_Change"] > 0 and \
    #                 has_traded == False and \
    #                 df.loc[i, "Close"] > df.loc[i, f"SMA_{timeperiod}"] and \
    #                 df.loc[i, "MACD"] < 0:
    #                 ask_price = df.loc[i, "Close"]
    #                 tp_price = ask_price + 0.0150
    #                 sl_price = ask_price - 0.0100
                    
    #                 local_order = {
    #                     "index": i,
    #                     "ask_price": ask_price,
    #                     "take_profit_price": tp_price, 
    #                     "stop_loss_price": sl_price, 
    #                     "position": 1,
    #                     f"SMA_{timeperiod}" : df.loc[i, f"SMA_{timeperiod}"],
    #                     "MACD" : df.loc[i, "MACD"],
    #                     'MACD_Signal' : df.loc[i, "MACD_Signal"],
    #                     "MACD_Hist" : df.loc[i, "MACD_Hist"],
    #                     "RSI" : df.loc[i, "RSI"],
    #                     "ATR" : df.loc[i, "ATR"],
    #                     "ADX" : df.loc[i, "ADX"],
    #                     "AROON_Oscillator" : df.loc[i, "AROON_Oscillator"],
    #                     "WILLR" : df.loc[i, "WILLR"],
    #                     "label": None,
    #                 }
    #                 has_traded = True
                
    #             if has_traded == True:
    #                 close_price = df.loc[i, "Close"]
    #                 if close_price >= tp_price:
    #                     outcome = 1
    #                     local_order["label"] = outcome
    #                     break
    #                 elif close_price <= sl_price:
    #                     outcome = 0
    #                     local_order["label"] = outcome
    #                     break
    #     else:
    #         for i in test_idx:
    #             if df.loc[i, "MACD_Crossover_Change"] < 0 and \
    #                 has_traded == False and \
    #                 df.loc[i, "Close"] < df.loc[i, f"SMA_{timeperiod}"] and \
    #                 df.loc[i, "MACD"] > 0:    
    #                 ask_price = df.loc[i, "Close"]
    #                 tp_price = ask_price - 0.0150
    #                 sl_price = ask_price + 0.0100
                    
    #                 local_order = {
    #                     "index": i,
    #                     "ask_price": ask_price,
    #                     "take_profit_price": tp_price, 
    #                     "stop_loss_price": sl_price, 
    #                     "position": 0,
    #                     f"SMA_{timeperiod}" : df.loc[i, f"SMA_{timeperiod}"],
    #                     "MACD" : df.loc[i, "MACD"],
    #                     'MACD_Signal' : df.loc[i, "MACD_Signal"],
    #                     "MACD_Hist" : df.loc[i, "MACD_Hist"],
    #                     "RSI" : df.loc[i, "RSI"],
    #                     "ATR" : df.loc[i, "ATR"],
    #                     "ADX" : df.loc[i, "ADX"],
    #                     "AROON_Oscillator" : df.loc[i, "AROON_Oscillator"],
    #                     "WILLR" : df.loc[i, "WILLR"],
    #                     "label": None,
    #                 }
    #                 has_traded = True
                
    #             if has_traded == True:
    #                 close_price = df.loc[i, "Close"]
    #                 if close_price <= tp_price:
    #                     outcome = 1
    #                     local_order["label"] = outcome
    #                     break
    #                 elif close_price >= sl_price:
    #                     outcome = 0
    #                     local_order["label"] = outcome
    #                     break
   
   
    return outcome, mape, local_order

split_y = splitter.split(y)
# create a parallel object with n_jobs processors
parallel = Parallel(n_jobs=-1)
# apply the forecast function to a range of indices in parallel
results = parallel(delayed(forecast_isolate)(j) for j, _ in enumerate(split_y))

# unpack the results into separate lists
outcomes, mapes, orders = zip(*results)
orders = list(orders)
orders_df = pd.DataFrame.from_dict(orders)
# drop the rows where any of the values is None
orders_df = orders_df.dropna()
# remove duplicate rows
orders_df = orders_df.drop_duplicates()
# remove duplicate indices
orders_df = orders_df.drop_duplicates(subset=['index'])
orders_df["year"] = year
orders_df["dataset_name"] = dataset_name

outcomes = [x for x in outcomes if x is not None]
outcomes_array = np.array(outcomes)
mapes_array = np.array(mapes)

accuracy_df = pd.DataFrame({
    'accuracy': accuracy_score([1] * len(outcomes), outcomes),
    'no_of_trades': len(outcomes),
    'no_of_wins': sum(outcomes),
    'no_of_losses': len(outcomes) - sum(outcomes),
    'slope_threshold': slope_threshold,
    'year': year,
    'step_length': step_length,
    "mean_mape": np.mean(mapes),
    "dataset_name": dataset_name,
    "algorithm": algorithm,
    "train_size": train_size,
    "fh": forecast_horizon,
    "window_size": window_size,
    'sma': timeperiod,
}, index=[0])


# # Save dataframe as a csv to output directory
accuracy_df.to_csv(f"results/{param_row}.csv", encoding='utf-8', index=False)
orders_df.to_csv(f"orders/{param_row}.csv", encoding='utf-8', index=False)
print("Done!!")
