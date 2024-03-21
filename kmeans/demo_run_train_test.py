import sys
import pandas as pd
import numpy as np
import talib
import warnings
import json
import matplotlib.pyplot as plt
import bisect
from sklearn.preprocessing import *
import mpl_toolkits.mplot3d
from sklearn.cluster import KMeans
import ffn as ffn
import empyrical as ep
from sktime.forecasting.model_selection import SlidingWindowSplitter

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

ONE_DAY = 4 * 24
INIT_CAPITAL = 100
N_CLOSE_PTS = int(param_dict["n_close_pts"])
N_PERC_PTS = int(param_dict["n_perc_pts"])
DIST_MEASURE = int(param_dict["dist_measure"])
N_CLUSTERS = int(param_dict["n_clusters"])
LOG_RETURN_THRESHOLD = param_dict["log_return_threshold"]
CALMAR_RATIO_THRESHOLD = param_dict["calmar_ratio_threshold"]
ATR_MULTIPLIER = int(param_dict["atr_multiplier"])
random_state = int(param_dict["random_state"])

# first_train_size = int(param_dict["first_train_size"] * ONE_DAY)
# second_train_size = int(param_dict["second_train_size"] * ONE_DAY)
# train_size = first_train_size + second_train_size

train_size = int(param_dict["train_size"] * ONE_DAY)
val_size = int(param_dict["val_size"] * ONE_DAY)
test_size = int(param_dict["test_size"] * ONE_DAY)
val_test_horizon = val_size + test_size


# Define a function to calculate the ulcer index
def m_ulcer_index(series):
    drawdown = (series - series.cummax()) / series.cummax()
    squared_average = (drawdown**2).mean()
    return squared_average**0.5

# DIST_MEASURE
# 1 = Euclidean Distance
# 2 = Perpendicular Distance
# 3 = Vertical Distance
def find_pips(data, n_pips):
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]]  # Price
    for curr_point in range(2, n_pips):
        md = 0.0  # Max distance
        md_i = -1  # Max distance index
        insert_index = -1
        # Use a single loop to iterate over all the points
        for i in range(1, len(data) - 1):
            left_adj = bisect.bisect_right(pips_x, i) - 1
            right_adj = left_adj + 1
            # Calculate the distance from the point to the line segment
            d = distance(data, pips_x, pips_y, i, left_adj, right_adj)
            # Update the maximum distance and the insert index
            if d > md:
                md = d
                md_i = i
                insert_index = right_adj
        # Insert the new pip
        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])
    return pips_x, pips_y

# Define a helper function to calculate the distance
def distance(data, pips_x, pips_y, i, left_adj, right_adj):
    time_diff = pips_x[right_adj] - pips_x[left_adj]
    price_diff = pips_y[right_adj] - pips_y[left_adj]
    slope = price_diff / time_diff
    intercept = pips_y[left_adj] - pips_x[left_adj] * slope
    dist_funcs = {
        1: lambda x, y: ((pips_x[left_adj] - x) ** 2 + (pips_y[left_adj] - y) ** 2)
        ** 0.5
        + ((pips_x[right_adj] - x) ** 2 + (pips_y[right_adj] - y) ** 2) ** 0.5,
        2: lambda x, y: abs((slope * x + intercept) - y) / (slope**2 + 1) ** 0.5,
        3: lambda x, y: abs((slope * x + intercept) - y),
    }
    return dist_funcs[DIST_MEASURE](i, data[i])

def get_pips_df(sub_df):
    pips_y_list = []
    # loop through the data
    for index in range(N_CLOSE_PTS, len(sub_df)):
        try:
            x_close = sub_df["log_close"].iloc[index - N_CLOSE_PTS : index].to_numpy()
            pips_x, pips_y = find_pips(x_close, N_PERC_PTS)
            scaled_pips_y = (
                StandardScaler()
                .fit_transform(np.array(pips_y).reshape(-1, 1))
                .reshape(-1)
            )
            pips_y_dict = {f"pip_{i}": scaled_pips_y[i] for i in range(N_PERC_PTS)}
            j = index - 1
            pips_y_dict["year"] = sub_df["year"].iloc[j]
            pips_y_dict["month"] = sub_df["month"].iloc[j]
            pips_y_dict["day_of_week"] = sub_df["day_of_week"].iloc[j]
            pips_y_dict["hour"] = sub_df["hour"].iloc[j]
            pips_y_dict["minute"] = sub_df["minute"].iloc[j]
            # future features
            tp = sub_df["log_close"].iloc[j] + (
                ATR_MULTIPLIER * sub_df["log_atr"].iloc[j]
            )
            sl = sub_df["log_close"].iloc[j] - (
                ATR_MULTIPLIER * sub_df["log_atr"].iloc[j]
            )
            for k in range(index, len(sub_df)):
                if sub_df["log_close"].iloc[k] >= tp:
                    pips_y_dict["future_return"] = 1
                    break
                elif sub_df["log_close"].iloc[k] <= sl:
                    pips_y_dict["future_return"] = -1
                    break
                else:
                    pips_y_dict["future_return"] = 0
            pips_y_list.append(pips_y_dict)
        except Exception as e:
            break
    pips_y_df = pd.DataFrame(pips_y_list)
    return pips_y_df

def cluster_and_filter_pips_df(pips_train_df):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state)
    kmeans.fit(
        pips_train_df[
            [
                "pip_0",
                "pip_1",
                "pip_2",
                "pip_3",
                "pip_4",
                "day_of_week",
                "hour",
                "minute",
            ]
        ].to_numpy()
    )
    pips_train_df["k_label"] = kmeans.labels_

 
    best_k_labels_list = []
    for k_label in np.arange(N_CLUSTERS):
        pips_y_sub_df = pips_train_df[(pips_train_df["k_label"] == k_label)]
        k_label_cumsum = pips_y_sub_df["future_return"].cumsum().reset_index(drop=True)
        if k_label_cumsum.iloc[-1] > 0:
            signal = 1
        else:
            signal = 0
            k_label_cumsum = -k_label_cumsum

        # Add a constant value to the series
        # Put INIT_CAPITAL as the first value
        start_portfolio = pd.concat(
            [pd.Series([INIT_CAPITAL]), (k_label_cumsum + INIT_CAPITAL)]
        ).reset_index(drop=True)
        
        if not start_portfolio.empty:
            start_k_label_cumsum = start_portfolio.iloc[0]
            end_k_label_cumsum = start_portfolio.iloc[-1]
        else:
            continue

        annualized_return = (end_k_label_cumsum / start_k_label_cumsum) - 1
        ulcer_index = m_ulcer_index(start_portfolio)
        max_drawdown = abs(ffn.calc_max_drawdown(start_portfolio)) + 0.001
        calmar_ratio = annualized_return / max_drawdown
        if (
            True
            and calmar_ratio > CALMAR_RATIO_THRESHOLD
            and annualized_return > LOG_RETURN_THRESHOLD
        ):

            best_k_labels_list.append(
                {
                    "signal": signal,
                    "k_label": k_label,
                    "calmar_ratio": calmar_ratio,
                    "ulcer_index": ulcer_index,
                    "annualized_return": annualized_return,
                    "max_drawdown": max_drawdown,
                    "actual_return": end_k_label_cumsum - start_k_label_cumsum,
                    "n_trades": len(k_label_cumsum),
                }
            )
    best_k_labels_df = pd.DataFrame(best_k_labels_list)
    return best_k_labels_df, kmeans

def filter_pips_df(pips_y_df, train_best_k_labels_df, kmeans):
    pips_y_df["k_label"] = kmeans.predict(pips_y_df[
        ["pip_0", "pip_1", "pip_2", "pip_3", "pip_4", "day_of_week", "hour", "minute"]
    ].to_numpy())

    test_k_labels_list = []

    for i in range(len(train_best_k_labels_df)):
        k_label = train_best_k_labels_df.iloc[i]["k_label"]
        signal = train_best_k_labels_df.iloc[i]["signal"]
        pips_y_copy_df = pips_y_df[(pips_y_df["k_label"] == k_label)]
        k_label_cumsum = pips_y_copy_df["future_return"].cumsum().reset_index(drop=True)
        if signal == 0:
            k_label_cumsum = -k_label_cumsum
        # Add a constant value to the series
        portfolio = pd.concat(
            [pd.Series([INIT_CAPITAL]), (k_label_cumsum + INIT_CAPITAL)]
        ).reset_index(drop=True)

        if not portfolio.empty:
            start_k_label_cumsum = portfolio.iloc[0]
            end_k_label_cumsum = portfolio.iloc[-1]
        else:
            continue

        annualized_return = (end_k_label_cumsum / start_k_label_cumsum) - 1
        ulcer_index = m_ulcer_index(portfolio)
        max_drawdown = abs(ffn.calc_max_drawdown(portfolio)) + 0.001
        calmar_ratio = annualized_return / max_drawdown

        test_k_labels_list.append(
            {
                "signal": signal,
                "k_label": k_label,
                "calmar_ratio": calmar_ratio,
                "ulcer_index": ulcer_index,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "actual_return": end_k_label_cumsum - start_k_label_cumsum,
                "n_trades": len(k_label_cumsum),
            }
        )
    return pd.DataFrame(test_k_labels_list)


ohlcv_data = pd.read_csv(
    # "/Users/newuser/Projects/robust_algo_trader/data/gen_oanda_data/GBP_USD_M15_raw_data.csv",
    "/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/GBP_USD_M15_raw_data.csv",
    parse_dates=["time"],
)
ohlcv_data = ohlcv_data.set_index("time")
ohlcv_data["year"] = ohlcv_data.index.year
ohlcv_data["month"] = ohlcv_data.index.month
ohlcv_data["day_of_week"] = ohlcv_data.index.dayofweek
ohlcv_data["hour"] = ohlcv_data.index.hour
ohlcv_data["minute"] = ohlcv_data.index.minute
ohlcv_data["log_close"] = np.log(ohlcv_data["close"])
ohlcv_data["log_high"] = np.log(ohlcv_data["high"])
ohlcv_data["log_low"] = np.log(ohlcv_data["low"])
ohlcv_data["log_atr"] = talib.ATR(ohlcv_data["log_high"], ohlcv_data["log_low"], ohlcv_data["log_close"], timeperiod=1)
start_date = "2007-01-01"
end_date = "2015-01-01"
ohlcv_data = ohlcv_data[start_date:end_date]
df = ohlcv_data.copy()

splitter = SlidingWindowSplitter(
    window_length=train_size,
    fh=np.arange(1, test_size + 1),
    step_length=test_size,
)

return_df_list = []
for i, (train_idx, test_idx) in enumerate(splitter.split(df)):        
    df_train = df.iloc[train_idx, :]
    df_test = df.iloc[test_idx, :]

    # TRAINING
    pips_train_df = get_pips_df(df_train)
    ts_scaler = StandardScaler().fit(pips_train_df[["day_of_week", "hour", "minute"]])
    pips_train_df[["day_of_week", "hour", "minute"]] = ts_scaler.transform(
        pips_train_df[["day_of_week", "hour", "minute"]]
    )
    train_best_k_labels_df, kmeans = cluster_and_filter_pips_df(pips_train_df)
    # ACCEPT OR REJECT THE TRAIN MODEL
    if train_best_k_labels_df.empty:
        continue

    # TESTING
    pips_test_df = get_pips_df(df_test)
    pips_test_df[["day_of_week", "hour", "minute"]] = ts_scaler.transform(
        pips_test_df[["day_of_week", "hour", "minute"]]
    )
    test_k_labels_df = filter_pips_df(pips_test_df, train_best_k_labels_df, kmeans)
    if test_k_labels_df.empty:
        continue

    return_df_list.append(
        {
            "window": i,
            "train_sum_annualized_return": train_best_k_labels_df["annualized_return"].sum(),
            "train_sum_actual_return": train_best_k_labels_df["actual_return"].sum(),
            "train_n_trades": train_best_k_labels_df["n_trades"].sum(),
            
            "test_sum_annualized_return": test_k_labels_df["annualized_return"].sum(),
            "test_sum_actual_return": test_k_labels_df["actual_return"].sum(),
            "test_n_trades": test_k_labels_df["n_trades"].sum(),
        }
    )
    if i >= 300:
        break
return_df = pd.DataFrame(return_df_list)
return_df["train_cumsum_annualized_return"] = return_df["train_sum_annualized_return"].cumsum()
return_df["train_cumsum_actual_return"] = return_df["train_sum_actual_return"].cumsum()
return_df["train_cumsum_n_trades"] = return_df["train_n_trades"].cumsum()

return_df["test_cumsum_annualized_return"] = return_df["test_sum_annualized_return"].cumsum()
return_df["test_cumsum_actual_return"] = return_df["test_sum_actual_return"].cumsum()
return_df["test_cumsum_n_trades"] = return_df["test_n_trades"].cumsum()


# return_df["n_close_pts"] = N_CLOSE_PTS
# return_df["n_perc_pts"] = N_PERC_PTS
# return_df["dist_measure"] = DIST_MEASURE
# return_df["n_clusters"] = N_CLUSTERS
# return_df["log_return_threshold"] = LOG_RETURN_THRESHOLD
# return_df["calmar_ratio_threshold"] = CALMAR_RATIO_THRESHOLD
return_df["train_size"] = train_size
return_df["test_size"] = test_size
return_df["random_state"] = random_state

out_file = f"results/{param_row}.csv"
return_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!")
