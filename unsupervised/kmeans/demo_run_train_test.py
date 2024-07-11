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
from sklearn.cluster import *
from sklearn.mixture import *
import ffn as ffn
import empyrical as ep
import joblib
from sktime.forecasting.model_selection import SlidingWindowSplitter
import numba

warnings.filterwarnings("ignore")

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1]) if len(sys.argv) == 2 else 0
param_dict = dict(params_df.iloc[param_row, :])

ONE_DAY = 4 * 24
INIT_CAPITAL = 100
RISK_FREE_RATE = 0.01

MAX_K_LABELS = int(param_dict["max_k_labels"])
N_CLOSE_PTS = int(param_dict["n_close_pts"])
N_PERC_PTS = int(param_dict["n_perc_pts"])
DIST_MEASURE = int(param_dict["dist_measure"])
N_CLUSTERS = int(param_dict["n_clusters"])
ATR_MULTIPLIER = int(param_dict["atr_multiplier"])
ALGORITHM = param_dict["algorithm"]

START_WINDOW_ITER = 40
MAX_WINDOW_ITER = 70

random_state = int(param_dict["random_state"])
train_size = int(param_dict["train_size"] * ONE_DAY)
test_size = int(param_dict["test_size"] * ONE_DAY)

estimators = {
    "kmeans": KMeans(n_clusters=N_CLUSTERS, random_state=random_state),
    "mini_batch_kmeans": MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=random_state
    ),
    "birch": Birch(n_clusters=N_CLUSTERS),
    "gaussian_mixture": GaussianMixture(
        n_components=N_CLUSTERS, covariance_type="tied", random_state=random_state
    ),
}


def calc_sharpe_ratio(portfolio_returns):
    excess_returns = np.array(portfolio_returns) - RISK_FREE_RATE
    standard_deviation = np.std(portfolio_returns)
    sharpe_ratio = np.mean(excess_returns) / standard_deviation
    return sharpe_ratio


def calc_sortino_ratio(portfolio_returns):
    excess_returns = np.array(portfolio_returns) - RISK_FREE_RATE
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns)
    sortino_ratio = np.mean(excess_returns) / downside_std
    return sortino_ratio


def calc_calmar_ratio(portfolio_returns):
    max_drawdown = ep.max_drawdown(portfolio_returns) + 0.001
    calmar_ratio = np.mean(portfolio_returns) / max_drawdown
    return calmar_ratio


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


def get_test_pips_df(sub_df, full_df, last_test_idx):
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

            if pips_y_dict["future_return"] == 0:
                # loop through the full_df starting from the last_test_idx
                # set future_return to 1 if tp is reached and -1 if sl is reached
                for k in range(last_test_idx, len(full_df)):
                    if full_df["log_close"].iloc[k] >= tp:
                        pips_y_dict["future_return"] = 1
                        break
                    elif full_df["log_close"].iloc[k] <= sl:
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
    pips_train_np = pips_train_df[
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
    estimator = estimators[ALGORITHM]
    estimator.fit(pips_train_np)
    pips_train_df["k_label"] = estimator.predict(pips_train_np)

    # group by k_label and calculate the cumulative sum of future returns
    filter_k_labels_df = (
        pips_train_df.groupby("k_label")["future_return"]
        .sum()
        .reset_index()
        .abs()
        .sort_values(by="future_return", ascending=False)
        .head(MAX_K_LABELS)
    )

    best_k_labels_list = []
    for k_label in filter_k_labels_df["k_label"]:
        pips_y_sub_df = pips_train_df[(pips_train_df["k_label"] == k_label)]
        k_label_cumsum = pips_y_sub_df["future_return"].cumsum().reset_index(drop=True)
        if k_label_cumsum.iloc[-1] > 0:
            signal = 1
        else:
            signal = 0
            k_label_cumsum = -k_label_cumsum

        # Add a constant value to the series
        # Put INIT_CAPITAL as the first value
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
    return best_k_labels_df, estimator


def filter_pips_df(pips_y_df, train_best_k_labels_df, estimator):
    pips_y_df["k_label"] = estimator.predict(
        pips_y_df[
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


def get_pips_df(sub_df):
    # Precompute necessary arrays
    close_array = sub_df["close"].to_numpy()
    high_array = sub_df["high"].to_numpy()
    low_array = sub_df["low"].to_numpy()
    atr_clipped_array = sub_df["atr_clipped"].to_numpy()

    pips_y_list = []
    scaler = StandardScaler()

    for index in range(N_CLOSE_PTS, len(sub_df)):
        x_close = close_array[max(0, index - N_CLOSE_PTS) : index]
        if len(x_close) < N_CLOSE_PTS:
            break  # Stop if we don't have enough historical data
        
        _, pips_y = find_pips(x_close, N_PERC_PTS)
        scaled_pips_y = (scaler
                         .fit_transform(np.array(pips_y).reshape(-1, 1))
                         .reshape(-1))
        pips_y_dict = {f"pip_{i}": scaled_pips_y[i] for i in range(N_PERC_PTS)}

        j = index - 1
        pips_y_dict.update(
            {
                "year": sub_df["year"].iloc[j],
                "month": sub_df["month"].iloc[j],
                "day_of_week": sub_df["day_of_week"].iloc[j],
                "hour": sub_df["hour"].iloc[j],
                "minute": sub_df["minute"].iloc[j],
            }
        )

        # Calculate future return
        tp = close_array[j] + (ATR_MULTIPLIER * atr_clipped_array[j])
        sl = close_array[j] - (ATR_MULTIPLIER * atr_clipped_array[j])

        future_highs = high_array[index:]
        future_lows = low_array[index:]

        if len(future_highs) > 0:
            tp_hit = np.argmax(future_highs >= tp)
            sl_hit = np.argmax(future_lows <= sl)

            # tp or sl were hit not by first value but by some other value
            # check if tp or sl has been hit first
            if tp_hit < sl_hit:
                pips_y_dict["future_return"] = 1  # TP hit first
            elif sl_hit < tp_hit:
                pips_y_dict["future_return"] = -1  # SL hit first
            elif future_highs[0] >= tp:
                pips_y_dict["future_return"] = 1  # TP hit
            elif future_lows[0] <= sl:
                pips_y_dict["future_return"] = -1  # SL hit
            # otherwise, then none of tp or sl were hit
            else:
                pips_y_dict["future_return"] = 0    
        else:
            pips_y_dict["future_return"] = 0  # No future data available
            
        pips_y_list.append(pips_y_dict)
    return pd.DataFrame(pips_y_list)


# Load the saved time scaler
ts_scaler = joblib.load("ts_scaler_2018.joblib")

# Read the CSV file
df = pd.read_csv(
    "/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/GBP_USD_M15_raw_data.csv",
    parse_dates=["time"],
    index_col="time",
)

# Extract date components efficiently
df["year"] = df.index.year
df["month"] = df.index.month
df["day_of_week"] = df.index.dayofweek
df["hour"] = df.index.hour
df["minute"] = df.index.minute

# Calculate ATR
df["atr"] = talib.ATR(
    df["high"].values, df["low"].values, df["close"].values, timeperiod=1
)
df["atr_clipped"] = np.clip(df["atr"], 0.00068, 0.00176)

# Filter date range
df = df.loc["2019-01-01":"2024-01-01"]

# Apply time scaling and rounding in one step
time_columns = ["day_of_week", "hour", "minute"]
df[time_columns] = np.round(ts_scaler.transform(df[time_columns]), 6)

# Round ATR columns
df[["atr", "atr_clipped"]] = df[["atr", "atr_clipped"]].round(6)


splitter = SlidingWindowSplitter(
    window_length=train_size,
    fh=np.arange(1, test_size + 1),
    step_length=test_size,
)

return_df_list = []
for i, (train_idx, test_idx) in enumerate(splitter.split(df)):
    # if i < START_WINDOW_ITER:
    #     continue

    df_train = df.iloc[train_idx, :]
    df_test = df.iloc[test_idx, :]
    last_test_idx = test_idx[-1]

    # TRAINING
    pips_train_df = get_pips_df(df_train)
    train_best_k_labels_df, estimator = cluster_and_filter_pips_df(pips_train_df)
    if train_best_k_labels_df.empty:
        continue

    # TESTING
    pips_test_df = get_test_pips_df(df_test, df, last_test_idx)
    test_k_labels_df = filter_pips_df(pips_test_df, train_best_k_labels_df, estimator)
    if test_k_labels_df.empty:
        continue

    return_df_list.append(
        {
            "window": i,
            "train_sum_annualized_return": train_best_k_labels_df[
                "annualized_return"
            ].sum(),
            "train_sum_actual_return": train_best_k_labels_df["actual_return"].sum(),
            "train_n_trades": train_best_k_labels_df["n_trades"].sum(),
            "test_sum_annualized_return": test_k_labels_df["annualized_return"].sum(),
            "test_sum_actual_return": test_k_labels_df["actual_return"].sum(),
            "test_n_trades": test_k_labels_df["n_trades"].sum(),
        }
    )
    # if i >= MAX_WINDOW_ITER:
    #     break
return_df = pd.DataFrame(return_df_list)
return_df["train_cumsum_annualized_return"] = return_df[
    "train_sum_annualized_return"
].cumsum()
return_df["train_cumsum_actual_return"] = return_df["train_sum_actual_return"].cumsum()
return_df["train_sharpe_ratio"] = calc_sharpe_ratio(
    return_df["train_sum_annualized_return"].to_numpy()
)

return_df["test_cumsum_annualized_return"] = return_df[
    "test_sum_annualized_return"
].cumsum()
return_df["test_cumsum_actual_return"] = return_df["test_sum_actual_return"].cumsum()

return_df["test_sharpe_ratio"] = calc_sharpe_ratio(
    return_df["test_sum_annualized_return"].to_numpy()
)
return_df["test_negative_sharpe_ratio"] = calc_sharpe_ratio(
    -1 * return_df["test_sum_annualized_return"].to_numpy()
)

# return_df["n_close_pts"] = N_CLOSE_PTS
# return_df["n_perc_pts"] = N_PERC_PTS
# return_df["dist_measure"] = DIST_MEASURE
return_df["max_k_labels"] = MAX_K_LABELS
return_df["n_clusters"] = N_CLUSTERS
return_df["algorithm"] = ALGORITHM
return_df["train_size"] = train_size
return_df["test_size"] = test_size
return_df["random_state"] = random_state

out_file = f"results/{param_row}.csv"
return_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!")
