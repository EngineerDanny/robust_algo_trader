import sys
import pandas as pd
import numpy as np
import talib as ta
import warnings
import json

warnings.filterwarnings("ignore")
params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

# Get the parameters for this task
param_dict = dict(params_df.iloc[param_row, :])
dataset_name = param_dict["dataset_name"]
strategy = param_dict["strategy"]
atr_delta = param_dict["atr_delta"]
train_end_date = param_dict["train_end_date"]

# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
    config = json.load(f)
config_settings = config["trading_settings"][dataset_name]
start_hr = config_settings["start_hour"]
end_hr = config_settings["end_hour"]
root_data_dir = config["paths"]["oanda_dir"]

df = pd.read_csv(f"{root_data_dir}/{dataset_name}_processed_data.csv")
df = df.rename(columns={"time": "Time"})
df["Index"] = df.index
df["Time"] = pd.to_datetime(df["Time"])
df = df[df["Time"] < train_end_date]
df["EMA_100"] = ta.EMA(df["Close"], timeperiod=100)
df = df.dropna()


def is_time_between(start_time, end_time, check_time):
    if start_time < end_time:
        return start_time <= check_time <= end_time
    else:
        return check_time >= start_time or check_time <= end_time

def macd_adaptive_profit_strategy(reverse=False):
    position = 0
    entry_price = 0
    stop_loss = 0
    trades = []
    for i, row in df.iterrows():
        current_time = row["Time"]
        condition_one = (
            (row["MACD_Crossover_Change"] > 0)
            and (row["Close"] > row["EMA_100"])
            and is_time_between(start_hr, end_hr, current_time.hour)
        )
        condition_two = (
            (row["MACD_Crossover_Change"] < 0)
            and (row["Close"] < row["EMA_100"])
            and is_time_between(start_hr, end_hr, current_time.hour)
        )
        condition_buy = condition_two if reverse else condition_one
        condition_sell = condition_one if reverse else condition_two
        if position == 0:
            if condition_buy:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - atr_delta * row["ATR"]
                init_stop_loss = stop_loss
                trades.append(["Buy", i, entry_price, 0])
            elif condition_sell:
                position = -1
                entry_price = row["Close"]
                stop_loss = entry_price + atr_delta * row["ATR"]
                init_stop_loss = stop_loss
                trades.append(["Sell", i, entry_price, 0])
        elif position == 1:
            if row["Low"] < stop_loss:
                position = 0
                exit_price = stop_loss
                target_profit = exit_price - entry_price
                target_loss = entry_price - init_stop_loss
                risk_reward = target_profit / target_loss
                trades[-1] = ["Buy", i, exit_price, risk_reward]
            else:
                stop_loss = max(stop_loss, row["Close"] - atr_delta * row["ATR"])
        elif position == -1:
            if row["High"] > stop_loss:
                position = 0
                exit_price = stop_loss
                target_profit = entry_price - exit_price
                target_loss = init_stop_loss - entry_price
                risk_reward = target_profit / target_loss
                trades[-1] = ["Sell", i, exit_price, risk_reward]
            else:
                stop_loss = min(stop_loss, row["Close"] + atr_delta * row["ATR"])
    trades_df = pd.DataFrame(trades, columns=["Action", "Date", "Price", "PnL"])
    trades_df["PnL_label"] = np.where(trades_df["PnL"] >= 0, 1, 0)
    return trades_df

def macd_fixed_profit_strategy(reverse=False):
    position = 0
    entry_price = 0
    stop_loss = 0
    trades = []
    for i, row in df.iterrows():
        current_time = row["Time"]
        condition_one = (
            (row["MACD_Crossover_Change"] > 0)
            and (row["Close"] > row["EMA_100"])
            and is_time_between(start_hr, end_hr, current_time.hour)
        )
        condition_two = (
            (row["MACD_Crossover_Change"] < 0)
            and (row["Close"] < row["EMA_100"])
            and is_time_between(start_hr, end_hr, current_time.hour)
        )
        condition_buy = condition_two if reverse else condition_one
        condition_sell = condition_one if reverse else condition_two
        if position == 0:
            if condition_buy:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - atr_delta * row["ATR"]
                take_profit = entry_price + atr_delta * row["ATR"]
                init_stop_loss = stop_loss
                trades.append(["Buy", i, entry_price, 0])
            elif condition_sell:
                position = -1
                entry_price = row["Close"]
                stop_loss = entry_price + atr_delta * row["ATR"]
                take_profit = entry_price - atr_delta * row["ATR"]
                init_stop_loss = stop_loss
                trades.append(["Sell", i, entry_price, 0])
        elif position == 1:
            if row["Low"] < stop_loss:
                position = 0
                exit_price = stop_loss
                target_profit = exit_price - entry_price
                target_loss = entry_price - init_stop_loss
                risk_reward = target_profit / target_loss
                trades[-1] = ["Buy", i, exit_price, risk_reward]
            elif row["High"] > take_profit:
                position = 0
                exit_price = take_profit
                target_profit = exit_price - entry_price
                target_loss = entry_price - init_stop_loss
                risk_reward = target_profit / target_loss
                trades[-1] = ["Buy", i, exit_price, risk_reward]
        elif position == -1:
            if row["High"] > stop_loss:
                position = 0
                exit_price = stop_loss
                target_profit = entry_price - exit_price
                target_loss = init_stop_loss - entry_price
                risk_reward = target_profit / target_loss
                trades[-1] = ["Sell", i, exit_price, risk_reward]
            elif row["Low"] < take_profit:
                position = 0
                exit_price = take_profit
                target_profit = entry_price - exit_price
                target_loss = init_stop_loss - entry_price
                risk_reward = target_profit / target_loss
                trades[-1] = ["Sell", i, exit_price, risk_reward]
    trades_df = pd.DataFrame(trades, columns=["Action", "Date", "Price", "PnL"])
    trades_df["PnL_label"] = np.where(trades_df["PnL"] >= 0, 1, 0)
    return trades_df


strategy_dict = {
    "MACD_Adaptive_Profit": macd_adaptive_profit_strategy,
    "Reverse_MACD_Adaptive_Profit": lambda: macd_adaptive_profit_strategy(reverse=True),
    "MACD_Fixed_Profit": macd_fixed_profit_strategy,
    "Reverse_MACD_Fixed_Profit": lambda: macd_fixed_profit_strategy(reverse=True),
}
trades_df = strategy_dict[strategy]()
# out_df should have the following columns: 
# dataset_name, strategy, atr_delta, accuracy(from PnL_label), num_trades
out_df = pd.DataFrame(
    {
        "dataset_name": dataset_name,
        "strategy": strategy,
        "atr_delta": atr_delta,
        "accuracy": trades_df["PnL_label"].mean(),
        "cummulative_pnl": trades_df["PnL"].sum(),
        "num_trades": trades_df.shape[0],
    },
    index=[0],
)
out_df.to_csv(f"results/{param_row}.csv", encoding='utf-8', index=False)
print("Done!")