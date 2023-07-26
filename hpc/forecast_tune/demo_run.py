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
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.model_selection import SlidingWindowSplitter
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

root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data" 
dataset_dict = {"EURUSD_H1" : f"{root_data_dir}/EURUSD_H1_200702210000_202304242100.tsv"}
dataset_path = dataset_dict[dataset_name]

def extract_df(path, timeperiod):
    real_df = pd.read_table(path)
    df = real_df.copy()
    df = df.drop(['<TICKVOL>', '<VOL>', '<SPREAD>'], axis=1)
    df = df.rename(columns={'<DATE>': 'Date', 
                                    '<TIME>': 'Time', 
                                    '<OPEN>': 'Open', 
                                    '<HIGH>': 'High', 
                                    '<LOW>': 'Low', 
                                    '<CLOSE>': 'Close'
                                    })
    # combine the date and time columns
    df['Date_Time'] = df['Date'] + ' ' + df['Time']
    # remove the date and time columns
    df = df.drop(['Date', 'Time'], axis=1)
    df['Time'] = pd.to_datetime(df['Date_Time'])
    df = df.drop(['Time'], axis=1)
    prices = df["Close"].values
    df["SMA"] = talib.SMA(prices, timeperiod=timeperiod)
    df = df.dropna()
    y = df[['SMA']]
    return y

y = extract_df(dataset_path, timeperiod)


learner_dict = {
    "LinearRegression": LinearRegression(),
    "LassoCV": LassoCV(random_state=1),
    'RidgeCV': RidgeCV(),
}

splitter = SlidingWindowSplitter(window_length=train_size, 
                                 fh=np.arange(1, forecast_horizon + 1), 
                                 step_length=24)
                                #  step_length=forecast_horizon)

mape_list = []
for j, (train_idx, test_idx) in enumerate(splitter.split(y)):
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    regressor = learner_dict[algorithm]
    forecaster = make_reduction(regressor, window_length=window_size, strategy="recursive")
    forecaster.fit(y_train)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    mape_list.append(mape)
    

test_err_df = pd.DataFrame({
    "mean_mape": np.mean(mape_list),
    "std_mape": np.std(mape_list),
    "train_size": train_size,
    "dataset_name": dataset_name,
    "algorithm": algorithm,
    "fh": forecast_horizon,
    "window_size": window_size,
    'sma': timeperiod,
}, index=[0])

# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
test_err_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")
