import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
from sklearn.linear_model import *
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sktime.forecasting.model_selection import SlidingWindowSplitter
from joblib import Parallel, delayed
from itertools import islice
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

root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/robust_algo_trader/data" 
dataset_dict = {
    "EURUSD_H1" : f"{root_data_dir}/EURUSD_H1_2007_2023_SMA_{timeperiod}.csv",
}
dataset_path = dataset_dict[dataset_name]

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

y = df[['SMA']]
offset = y.index[0]


def forecast_isolate(j):
    outcome = None
    
    # get the train and test indices
    splitter_y = splitter.split(y)
    train_idx, test_idx = next(islice(splitter_y, j, None))

    # train_idx, test_idx = splitter.split(y)[j]
    train_idx = train_idx + offset
    test_idx = test_idx + offset
    
    # get the train and test data
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # create a forecaster object using LinearRegression and recursive strategy
    regressor = learner_dict[algorithm]
    forecaster = make_reduction(regressor, 
                                window_length=window_size, 
                                strategy="recursive")
    forecaster.fit(y_train)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    
    x = np.arange(1, len(y_pred) + 1)
    slope, intercept = np.polyfit(x, y_pred, 1)
    y_fit = np.polyval([slope, intercept], x)
    mse = mean_squared_error(y_pred, y_fit)
    mse_scaled = mse * 10e8
    slope_scaled = slope * 10e5
    
    if abs(slope_scaled) > slope_threshold:
        has_traded = False
        if slope > 0:
            for i in test_idx:
                # check if there is a buy MACD crossover
                if df.loc[i, "MACD_Crossover_Change"] > 0 and \
                    has_traded == False and \
                    df.loc[i, "Close"] > df.loc[i, "SMA"] and \
                    df.loc[i, "MACD"] < 0:
                    ask_price = df.loc[i, "Close"]
                    tp_price = ask_price + 0.0150
                    sl_price = ask_price - 0.0100
                    
                    order = {
                        "ask_price": ask_price,
                        "take_profit_price": tp_price, 
                        "stop_loss_price": sl_price, 
                        "position": "long",
                        "MACD" : df.loc[i, "MACD"],
                        "SMA" : df.loc[i, "SMA"],
                    }
                    orders.append(order)
                    has_traded = True
                
                if has_traded == True:
                    close_price = df.loc[i, "Close"]
                    if close_price >= tp_price:
                        outcome = 1
                        break
                    elif close_price <= sl_price:
                        outcome = 0
                        break
        else:
            for i in test_idx:
                if df.loc[i, "MACD_Crossover_Change"] < 0 and \
                    has_traded == False and \
                    df.loc[i, "Close"] < df.loc[i, "SMA"] and \
                    df.loc[i, "MACD"] > 0:    
                    ask_price = df.loc[i, "Close"]
                    tp_price = ask_price - 0.0150
                    sl_price = ask_price + 0.0100
                    
                    order = {
                        "ask_price": ask_price,
                        "take_profit_price": tp_price, 
                        "stop_loss_price": sl_price, 
                        "position": "short",
                    }
                    orders.append(order)
                    has_traded = True
                
                if has_traded == True:
                    close_price = df.loc[i, "Close"]
                    if close_price <= tp_price:
                        outcome = 1
                        break
                    elif close_price >= sl_price:
                        outcome = 0
                        break

    return outcome, mape

split_y = splitter.split(y)
# create a parallel object with n_jobs processors
parallel = Parallel(n_jobs=-1)
# apply the forecast function to a range of indices in parallel
results = parallel(delayed(forecast_isolate)(j) for j, _ in enumerate(split_y))
# results = parallel(delayed(forecast_isolate)(j) for j in range(5))

# unpack the results into separate lists
outcomes, mapes = zip(*results)
outcomes = [x for x in outcomes if x is not None]

outcomes_array = np.array(outcomes)
mapes_array = np.array(mapes)


# apply the forecast_isolate function to the split_y list in parallel
# results = parallel.apply_async(forecast_isolate, split_y)

# get the results as a list
# results = results.get()

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


# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
accuracy_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")
