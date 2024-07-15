import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import SlidingWindowSplitter
import os
import sys
import talib
import warnings
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from numba import jit
import joblib
sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/utils.py"))
from utils import *



warnings.filterwarnings("ignore")

# Constants
CANDLES_PER_DAY = 4 * 24  # 15-minute candles
INITIAL_CAPITAL = 100
RISK_FREE_RATE = 0.01

# Load trading parameters from CSV
trading_params = pd.read_csv("params.csv")
param_row = 0 if len(sys.argv) != 2 else int(sys.argv[1])
param_dict = dict(
    trading_params.iloc[param_row, :]
)  # Assume first row of trading_params.csv

# Extract trading parameters
MAX_CLUSTER_LABELS = int(param_dict["max_cluster_labels"])
PRICE_HISTORY_LENGTH = int(param_dict["price_history_length"])
NUM_PERCEPTUALLY_IMPORTANT_POINTS = int(param_dict["num_perceptually_important_points"])
DISTANCE_MEASURE = int(param_dict["distance_measure"])
NUM_CLUSTERS = int(param_dict["num_clusters"])
ATR_MULTIPLIER = int(param_dict["atr_multiplier"])
CLUSTERING_ALGORITHM = param_dict["clustering_algorithm"]
RANDOM_SEED = int(param_dict["random_seed"])
TRAIN_PERIOD = int(param_dict["train_period"] * CANDLES_PER_DAY)
TEST_PERIOD = int(param_dict["test_period"] * CANDLES_PER_DAY)

# Define clustering algorithms
clustering_models = {
    "kmeans": KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED),
    "gaussian_mixture": GaussianMixture(
        n_components=NUM_CLUSTERS, covariance_type="tied", random_state=RANDOM_SEED
    ),
}


@jit(nopython=True)
def calculate_sharpe_ratio(returns):
    excess_returns = returns - RISK_FREE_RATE
    return np.mean(excess_returns) / np.std(returns)


@jit(nopython=True)
def find_perceptually_important_points(price_data, num_points):
    point_indices = np.zeros(num_points, dtype=np.int64)
    point_prices = np.zeros(num_points, dtype=np.float64)
    point_indices[0], point_indices[1] = 0, len(price_data) - 1
    point_prices[0], point_prices[1] = price_data[0], price_data[-1]

    for current_point in range(2, num_points):
        max_distance, max_distance_index, insert_index = 0.0, -1, -1
        for i in range(1, len(price_data) - 1):
            left_adj = (
                np.searchsorted(point_indices[:current_point], i, side="right") - 1
            )
            right_adj = left_adj + 1
            distance = calculate_point_distance(
                price_data,
                point_indices[:current_point],
                point_prices[:current_point],
                i,
                left_adj,
                right_adj,
            )
            if distance > max_distance:
                max_distance, max_distance_index, insert_index = distance, i, right_adj

        point_indices[insert_index + 1 : current_point + 1] = point_indices[
            insert_index:current_point
        ]
        point_prices[insert_index + 1 : current_point + 1] = point_prices[
            insert_index:current_point
        ]
        point_indices[insert_index], point_prices[insert_index] = (
            max_distance_index,
            price_data[max_distance_index],
        )

    return point_indices, point_prices


@jit(nopython=True)
def calculate_point_distance(
    data, point_indices, point_prices, index, left_adj, right_adj
):
    time_diff = point_indices[right_adj] - point_indices[left_adj]
    price_diff = point_prices[right_adj] - point_prices[left_adj]
    slope = price_diff / time_diff
    intercept = point_prices[left_adj] - point_indices[left_adj] * slope
    x, y = index, data[index]

    if DISTANCE_MEASURE == 1:
        return (
            (point_indices[left_adj] - x) ** 2 + (point_prices[left_adj] - y) ** 2
        ) ** 0.5 + (
            (point_indices[right_adj] - x) ** 2 + (point_prices[right_adj] - y) ** 2
        ) ** 0.5
    elif DISTANCE_MEASURE == 2:
        return abs((slope * x + intercept) - y) / (slope**2 + 1) ** 0.5
    else:  # DISTANCE_MEASURE == 3
        return abs((slope * x + intercept) - y)


@jit(nopython=True)
def determine_trade_outcome(future_highs, future_lows, take_profit, stop_loss):   
    """
    Determine the outcome of a trade based on future high and low prices and TP/SL levels.

    Args:
    future_highs (np.array): Array of future high prices.
    future_lows (np.array): Array of future low prices.
    take_profit (float): Take profit level.
    stop_loss (float): Stop loss level.

    Returns:
    int: 1 for profit, -1 for loss, 0 for no action.
    """    
    # Check if the first value hits TP or SL
    if future_highs[0] >= take_profit:
        return 1
    if future_lows[0] <= stop_loss:
        return -1
    
    tp_hit = np.argmax(future_highs >= take_profit)
    sl_hit = np.argmax(future_lows <= stop_loss)

    if tp_hit == 0 and sl_hit == 0:
        return 0
    elif (tp_hit < sl_hit and tp_hit != 0) or (tp_hit != 0 and sl_hit == 0):
        return 1
    elif (sl_hit < tp_hit and sl_hit != 0) or (sl_hit != 0 and tp_hit == 0):
        return -1
    else:
        return 0


def prepare_test_data(price_subset, full_price_data, last_test_index):
    test_data_list = []
    scaler = StandardScaler()

    for index in range(PRICE_HISTORY_LENGTH, len(price_subset)):
        price_history = (
            price_subset["close"].iloc[index - PRICE_HISTORY_LENGTH : index].values
        )
        if len(price_history) < PRICE_HISTORY_LENGTH:
            continue

        _, important_points = find_perceptually_important_points(
            price_history, NUM_PERCEPTUALLY_IMPORTANT_POINTS
        )
        scaled_points = scaler.fit_transform(important_points.reshape(-1, 1)).flatten()

        data_point = {
            f"price_point_{i}": scaled_points[i]
            for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        }
        data_point.update(
            price_subset.iloc[index - 1][
                ["year", "month", "day_of_week", "hour", "minute"]
            ].to_dict()
        )

        current_price = price_subset["close"].iloc[index - 1]
        current_atr = price_subset["atr_clipped"].iloc[index - 1]
        take_profit = current_price + (ATR_MULTIPLIER * current_atr)
        stop_loss = current_price - (ATR_MULTIPLIER * current_atr)

        future_highs = price_subset["high"].iloc[index:].values
        future_lows = price_subset["low"].iloc[index:].values

        data_point["trade_outcome"] = determine_trade_outcome(
            future_highs, future_lows, take_profit, stop_loss
        )
        if data_point["trade_outcome"] == 0:
            future_highs_full = full_price_data["high"].iloc[last_test_index:].values
            future_lows_full = full_price_data["low"].iloc[last_test_index:].values
            data_point["trade_outcome"] = determine_trade_outcome(
                future_highs_full, future_lows_full, take_profit, stop_loss
            )

        test_data_list.append(data_point)

    return pd.DataFrame(test_data_list)


# Define the dtype for our structured array
performance_dtype = np.dtype(
    [
        ("signal", np.int64),
        ("cluster_label", np.int64),
        ("calmar_ratio", np.float64),
        ("annualized_return", np.float64),
        ("max_drawdown", np.float64),
        ("actual_return", np.float64),
        ("num_trades", np.int64),
    ]
)


def evaluate_cluster_performance_df(
    price_data_df, train_best_clusters_df, clustering_model
):
    """
    Evaluate cluster performance and return results as a DataFrame.

    Args:
    price_data_df (pd.DataFrame): DataFrame of price data.
    train_best_clusters_df (pd.DataFrame): DataFrame of best performing clusters from training.
    clustering_model (object): Trained clustering model.

    Returns:
    pd.DataFrame: DataFrame of cluster performance metrics.
    """
    price_data = price_data_df[
        [
            "price_point_0",
            "price_point_1",
            "price_point_2",
            "price_point_3",
            "price_point_4",
            "day_of_week",
            "hour",
            "minute",
            "trade_outcome",
        ]
    ].values
    train_best_clusters = train_best_clusters_df[["cluster_label", "signal"]].values

    # Predict cluster labels and evaluate performance
    # Remove the trade_outcome column from the price data before
    predicted_labels = clustering_model.predict(price_data[:, :-1])
    cluster_performance_list = np.zeros(
        len(train_best_clusters), dtype=performance_dtype
    )

    for i in range(len(train_best_clusters)):
        cluster_label, signal = train_best_clusters[i]
        mask = predicted_labels == cluster_label
        cluster_cumulative_return = np.cumsum(price_data[mask, -1])
        if signal == 0:
            cluster_cumulative_return = -cluster_cumulative_return

        cluster_trade_outcomes = price_data[mask, -1]
        metrics = calculate_evaluation_metrics(
            cluster_cumulative_return, cluster_trade_outcomes
        )
        cluster_performance_list[i]["signal"] = signal
        cluster_performance_list[i]["cluster_label"] = cluster_label
        cluster_performance_list[i]["calmar_ratio"] = metrics[0]
        cluster_performance_list[i]["annualized_return"] = metrics[1]
        cluster_performance_list[i]["max_drawdown"] = metrics[2]
        cluster_performance_list[i]["actual_return"] = metrics[3]
        cluster_performance_list[i]["num_trades"] = metrics[4]

    return pd.DataFrame(cluster_performance_list)


@jit(nopython=True)
def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown of a portfolio.

    Args:
    portfolio_values (np.array): Array of portfolio values over time.

    Returns:
    float: Maximum drawdown as a percentage.
    """
    peak = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values[1:]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


@jit(nopython=True)
def calculate_trading_metrics(trade_outcomes):
    """
    Calculate trading metrics based on trade outcomes.

    Args:
    trade_outcomes (np.array): Array of trade outcomes.

    Returns:
    tuple: Calculated trading metrics.
    """
    if len(trade_outcomes) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0

    cumulative_return = np.cumsum(trade_outcomes)
    signal = 1 if cumulative_return[-1] > 0 else 0
    if signal == 0:
        cumulative_return = -cumulative_return

    portfolio_values = np.zeros(len(cumulative_return) + 1)
    portfolio_values[0] = INITIAL_CAPITAL
    portfolio_values[1:] = cumulative_return + INITIAL_CAPITAL

    start_value, end_value = portfolio_values[0], portfolio_values[-1]
    annualized_return = (end_value / start_value) - 1
    max_drawdown = calculate_max_drawdown(portfolio_values)
    calmar_ratio = annualized_return / (max_drawdown + 1e-6)
    actual_return = end_value - start_value
    return (
        signal,
        calmar_ratio,
        annualized_return,
        max_drawdown,
        actual_return,
        len(trade_outcomes),
    )


@jit(nopython=True)
def calculate_evaluation_metrics(cumulative_return, trade_outcomes):
    if len(trade_outcomes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    portfolio_values = np.zeros(len(cumulative_return) + 1)
    portfolio_values[0] = INITIAL_CAPITAL
    portfolio_values[1:] = cumulative_return + INITIAL_CAPITAL

    start_value, end_value = portfolio_values[0], portfolio_values[-1]
    annualized_return = (end_value / start_value) - 1
    max_drawdown = calculate_max_drawdown(portfolio_values)
    calmar_ratio = annualized_return / (max_drawdown + 1e-6)
    actual_return = end_value - start_value
    
    return (
        calmar_ratio,
        annualized_return,
        max_drawdown,
        actual_return,
        len(trade_outcomes),
    )


def cluster_and_evaluate_price_data(price_data_df):
    price_features = price_data_df[
        [
            "price_point_0",
            "price_point_1",
            "price_point_2",
            "price_point_3",
            "price_point_4",
            "day_of_week",
            "hour",
            "minute",
        ]
    ].values
    clustering_model = clustering_models[CLUSTERING_ALGORITHM]
    clustering_model.fit(price_features)
    price_data_df["cluster_label"] = clustering_model.predict(price_features)

    top_clusters_df = (
        price_data_df.groupby("cluster_label")["trade_outcome"]
        .sum()
        .abs()
        .nlargest(MAX_CLUSTER_LABELS)
        .reset_index()
    )

    best_clusters_list = []
    for cluster_label in top_clusters_df["cluster_label"]:
        cluster_trade_outcomes = price_data_df[
            price_data_df["cluster_label"] == cluster_label
        ]["trade_outcome"].values
        metrics = calculate_trading_metrics(cluster_trade_outcomes)
        best_clusters_list.append(
            {
                "signal": metrics[0],
                "cluster_label": cluster_label,
                "calmar_ratio": metrics[1],
                "annualized_return": metrics[2],
                "max_drawdown": metrics[3],
                "actual_return": metrics[4],
                "num_trades": metrics[5],
            }
        )

    return pd.DataFrame(best_clusters_list), clustering_model


def prepare_training_data(price_subset):
    training_data_list = []
    scaler = StandardScaler()

    for index in range(PRICE_HISTORY_LENGTH, len(price_subset)):
        price_history = (
            price_subset["close"]
            .iloc[max(0, index - PRICE_HISTORY_LENGTH) : index]
            .values
        )
        if len(price_history) < PRICE_HISTORY_LENGTH:
            break

        _, important_points = find_perceptually_important_points(
            price_history, NUM_PERCEPTUALLY_IMPORTANT_POINTS
        )
        scaled_points = scaler.fit_transform(important_points.reshape(-1, 1)).flatten()

        data_point = {
            f"price_point_{i}": scaled_points[i]
            for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        }
        data_point.update(
            price_subset.iloc[index - 1][
                ["year", "month", "day_of_week", "hour", "minute"]
            ].to_dict()
        )

        current_price = price_subset["close"].iloc[index - 1]
        current_atr = price_subset["atr_clipped"].iloc[index - 1]
        take_profit = current_price + (ATR_MULTIPLIER * current_atr)
        stop_loss = current_price - (ATR_MULTIPLIER * current_atr)

        future_highs = price_subset["high"].iloc[index:].values
        future_lows = price_subset["low"].iloc[index:].values

        if len(future_highs) > 0:
            data_point["trade_outcome"] = determine_trade_outcome(
                future_highs, future_lows, take_profit, stop_loss
            )
        else:
            data_point["trade_outcome"] = 0

        training_data_list.append(data_point)

    return pd.DataFrame(training_data_list)


def main():
    time_scaler = joblib.load("/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/ts_scaler_2018.joblib")
    price_data = pd.read_csv(
        "/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/GBP_USD_M15_raw_data.csv",
        parse_dates=["time"],
        index_col="time",
    )

    price_data["year"] = price_data.index.year
    price_data["month"] = price_data.index.month
    price_data["day_of_week"] = price_data.index.dayofweek
    price_data["hour"] = price_data.index.hour
    price_data["minute"] = price_data.index.minute
    price_data["atr"] = talib.ATR(
        price_data["high"].values,
        price_data["low"].values,
        price_data["close"].values,
        timeperiod=1,
    )
    price_data["atr_clipped"] = np.clip(price_data["atr"], 0.00068, 0.00176)

    # Filter date range and apply time scaling
    price_data = price_data.loc["2019-01-01":"2024-05-01"]
    time_columns = ["day_of_week", "hour", "minute"]
    price_data[time_columns] = np.round(
        time_scaler.transform(price_data[time_columns]), 6
    )
    price_data[["atr", "atr_clipped"]] = price_data[["atr", "atr_clipped"]].round(6)

    # Initialize the sliding window splitter for backtesting
    window_splitter = RandomStartSlidingWindowSplitter(
        train_size=TRAIN_PERIOD,
        test_size=TEST_PERIOD,
        n_splits=50, 
        randomness=0.5
    )

    backtest_results = []
    for window, (train_indices, test_indices) in enumerate(
        window_splitter.split(price_data)
    ):
        print(f"Processing window {window}...")
        train_data = price_data.iloc[train_indices, :]
        test_data = price_data.iloc[test_indices, :]
        last_test_index = test_indices[-1]

        # Prepare training data and perform clustering
        print("Preparing training data and clustering...")
        train_price_data = prepare_training_data(train_data)
        train_best_clusters, clustering_model = cluster_and_evaluate_price_data(
            train_price_data
        )
        if train_best_clusters.empty:
            continue

        # Prepare test data and evaluate cluster performance
        print("Preparing test data and evaluating cluster performance...")
        test_price_data = prepare_test_data(test_data, price_data, last_test_index)
        test_cluster_performance = evaluate_cluster_performance_df(
            test_price_data, train_best_clusters, clustering_model
        )
        if test_cluster_performance.empty:
            continue

        # Compile results for this window
        print("Compiling results...")
        window_result = {
            "window": window,
            "train_total_annualized_return": train_best_clusters[
                "annualized_return"
            ].sum(),
            "train_total_actual_return": train_best_clusters["actual_return"].sum(),
            "train_total_trades": train_best_clusters["num_trades"].sum(),
            "test_total_annualized_return": test_cluster_performance[
                "annualized_return"
            ].sum(),
            "test_total_actual_return": test_cluster_performance["actual_return"].sum(),
            "test_total_trades": test_cluster_performance["num_trades"].sum(),
        }
        backtest_results.append(window_result)

    # Compile final results
    results_df = pd.DataFrame(backtest_results)
    results_df["train_cumulative_annualized_return"] = results_df[
        "train_total_annualized_return"
    ].cumsum()
    results_df["train_cumulative_actual_return"] = results_df[
        "train_total_actual_return"
    ].cumsum()
    results_df["train_sharpe_ratio"] = calculate_sharpe_ratio(
        results_df["train_total_annualized_return"].values
    )

    results_df["test_cumulative_annualized_return"] = results_df[
        "test_total_annualized_return"
    ].cumsum()
    results_df["test_cumulative_actual_return"] = results_df[
        "test_total_actual_return"
    ].cumsum()
    results_df["test_sharpe_ratio"] = calculate_sharpe_ratio(
        results_df["test_total_annualized_return"].values
    )
    results_df["test_inverse_sharpe_ratio"] = calculate_sharpe_ratio(
        -1 * results_df["test_total_annualized_return"].values
    )

    # Add constant parameters to the results
    results_df["max_cluster_labels"] = MAX_CLUSTER_LABELS
    results_df["num_clusters"] = NUM_CLUSTERS
    results_df["clustering_algorithm"] = CLUSTERING_ALGORITHM
    results_df["train_period"] = TRAIN_PERIOD
    results_df["test_period"] = TEST_PERIOD
    results_df["random_seed"] = RANDOM_SEED

    # save results to csv
    out_file = f"results/{param_row}.csv"
    results_df.to_csv(out_file, encoding="utf-8", index=False)
    print("Backtesting completed.")

if __name__ == "__main__":
    main()
