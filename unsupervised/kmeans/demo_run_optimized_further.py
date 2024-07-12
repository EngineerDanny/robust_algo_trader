import sys
import numpy as np
import pandas as pd
import talib
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture
from sktime.forecasting.model_selection import SlidingWindowSplitter
from numba import jit
import joblib

warnings.filterwarnings("ignore")

# Constants
CANDLES_PER_DAY = 4 * 24  # 15-minute candles
INITIAL_CAPITAL = 100
RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252

# Load trading parameters from CSV
trading_params = pd.read_csv("params.csv")
param_row = 0 if len(sys.argv) != 2 else int(sys.argv[1])
param_dict = dict(trading_params.iloc[param_row, :])

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
    "mini_batch_kmeans": MiniBatchKMeans(
        n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED
    ),
    "birch": Birch(n_clusters=NUM_CLUSTERS),
    "gaussian_mixture": GaussianMixture(
        n_components=NUM_CLUSTERS, covariance_type="tied", random_state=RANDOM_SEED
    ),
}


@jit(nopython=True)
def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    avg_excess_return = np.mean(excess_returns)
    std_dev = np.std(returns)
    if std_dev == 0:
        return 0.0 if avg_excess_return == 0 else np.inf * np.sign(avg_excess_return)
    return np.sqrt(TRADING_DAYS_PER_YEAR) * avg_excess_return / std_dev


@jit(nopython=True)
def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    avg_excess_return = np.mean(excess_returns)
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return 0.0 if avg_excess_return == 0 else np.inf * np.sign(avg_excess_return)
    return np.sqrt(TRADING_DAYS_PER_YEAR) * avg_excess_return / downside_deviation


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
    if future_highs[0] >= take_profit:
        return 1
    if future_lows[0] <= stop_loss:
        return -1

    tp_hit = np.argmax(future_highs >= take_profit)
    sl_hit = np.argmax(future_lows <= stop_loss)

    if tp_hit == 0 and sl_hit == 0:
        return 0
    elif tp_hit < sl_hit or (tp_hit > 0 and sl_hit == 0):
        return 1
    elif sl_hit < tp_hit or (sl_hit > 0 and tp_hit == 0):
        return -1
    else:
        return 0


@jit(nopython=True)
def calculate_max_drawdown(portfolio_values):
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
    if len(trade_outcomes) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0

    cumulative_return = np.cumprod(1 + trade_outcomes) - 1
    total_return = cumulative_return[-1]

    portfolio_values = INITIAL_CAPITAL * (1 + cumulative_return)

    max_drawdown = calculate_max_drawdown(portfolio_values)
    sharpe_ratio = calculate_sharpe_ratio(trade_outcomes)
    sortino_ratio = calculate_sortino_ratio(trade_outcomes)

    return (
        1 if total_return > 0 else 0,  # signal
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        total_return * INITIAL_CAPITAL,  # actual return in currency units
        len(trade_outcomes),
    )


@jit(nopython=True)
def predict_clusters(price_data, cluster_centers):
    num_samples = price_data.shape[0]
    num_clusters = cluster_centers.shape[0]
    distances = np.zeros((num_samples, num_clusters))

    for i in range(num_samples):
        for j in range(num_clusters):
            distances[i, j] = np.sum((price_data[i] - cluster_centers[j]) ** 2)

    return np.argmin(distances, axis=1)


@jit(nopython=True)
def evaluate_cluster_performance(price_data, best_clusters, cluster_centers):
    predicted_labels = predict_clusters(price_data[:, :8], cluster_centers)
    num_clusters = len(best_clusters)

    cluster_performance_list = np.zeros((num_clusters, 7), dtype=np.float64)

    for i in range(num_clusters):
        cluster_label, signal = best_clusters[i]
        mask = predicted_labels == cluster_label
        cluster_returns = price_data[mask, -1]

        if signal == 0:
            cluster_returns = -cluster_returns

        if len(cluster_returns) > 0:
            metrics = calculate_trading_metrics(cluster_returns)

            cluster_performance_list[i, 0] = signal
            cluster_performance_list[i, 1] = cluster_label
            cluster_performance_list[i, 2] = metrics[1]  # sharpe_ratio
            cluster_performance_list[i, 3] = metrics[2]  # sortino_ratio
            cluster_performance_list[i, 4] = metrics[3]  # max_drawdown
            cluster_performance_list[i, 5] = metrics[4]  # actual_return
            cluster_performance_list[i, 6] = metrics[5]  # num_trades
        else:
            # If there are no returns for this cluster, set all metrics to 0
            cluster_performance_list[i] = [signal, cluster_label, 0, 0, 0, 0, 0]

    return cluster_performance_list


def evaluate_cluster_performance_df(
    price_data_df, train_best_clusters_df, clustering_model
):
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

    cluster_centers = clustering_model.cluster_centers_

    result = evaluate_cluster_performance(
        price_data, train_best_clusters, cluster_centers
    )

    df = pd.DataFrame(
        result,
        columns=[
            "signal",
            "cluster_label",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "actual_return",
            "num_trades",
        ],
    )

    # Remove rows where all metrics are 0 (no trades in that cluster)
    df = df[
        (df["sharpe_ratio"] != 0)
        | (df["sortino_ratio"] != 0)
        | (df["max_drawdown"] != 0)
        | (df["actual_return"] != 0)
        | (df["num_trades"] != 0)
    ]

    return df


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
        current_atr = price_subset["atr"].iloc[index - 1]
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
                "sharpe_ratio": metrics[1],
                "sortino_ratio": metrics[2],
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
    time_scaler = joblib.load("ts_scaler_2018.joblib")

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
    price_data = price_data.loc["2019-01-01":"2019-05-01"]
    time_columns = ["day_of_week", "hour", "minute"]
    price_data[time_columns] = np.round(
        time_scaler.transform(price_data[time_columns]), 6
    )
    price_data[["atr", "atr_clipped"]] = price_data[["atr", "atr_clipped"]].round(6)

    # Initialize the sliding window splitter for backtesting
    window_splitter = SlidingWindowSplitter(
        window_length=TRAIN_PERIOD,
        fh=np.arange(1, TEST_PERIOD + 1),
        step_length=TEST_PERIOD,
    )

    backtest_results = []
    cumulative_return = 1.0
    all_returns = []

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
            print(f"No valid clusters in training window {window}, skipping...")
            continue

        # Prepare test data and evaluate cluster performance
        print("Preparing test data and evaluating cluster performance...")
        test_price_data = prepare_test_data(test_data, price_data, last_test_index)
        test_cluster_performance = evaluate_cluster_performance_df(
            test_price_data, train_best_clusters, clustering_model
        )
        if test_cluster_performance.empty:
            print(f"No valid trades in test window {window}, skipping...")
            continue

        # Calculate window returns
        train_return = train_best_clusters["actual_return"].sum() / INITIAL_CAPITAL
        test_return = test_cluster_performance["actual_return"].sum() / INITIAL_CAPITAL

        all_returns.append(test_return)
        cumulative_return *= 1 + test_return

        # Compile results for this window
        print("Compiling results...")
        window_result = {
            "window": window,
            "train_return": train_return,
            "train_sharpe_ratio": calculate_sharpe_ratio(
                train_best_clusters["actual_return"].values / INITIAL_CAPITAL
            ),
            "train_sortino_ratio": calculate_sortino_ratio(
                train_best_clusters["actual_return"].values / INITIAL_CAPITAL
            ),
            "train_total_trades": train_best_clusters["num_trades"].sum(),
            "test_return": test_return,
            "test_sharpe_ratio": calculate_sharpe_ratio(
                test_cluster_performance["actual_return"].values / INITIAL_CAPITAL
            ),
            "test_sortino_ratio": calculate_sortino_ratio(
                test_cluster_performance["actual_return"].values / INITIAL_CAPITAL
            ),
            "test_total_trades": test_cluster_performance["num_trades"].sum(),
            "cumulative_return": cumulative_return - 1,  # Convert to percentage
        }
        backtest_results.append(window_result)

    # Compile final results
    results_df = pd.DataFrame(backtest_results)

    # Calculate overall metrics
    total_days = len(price_data)
    years = total_days / (CANDLES_PER_DAY * TRADING_DAYS_PER_YEAR)

    overall_annualized_return = (cumulative_return ** (1 / years)) - 1
    overall_sharpe_ratio = calculate_sharpe_ratio(np.array(all_returns))
    overall_sortino_ratio = calculate_sortino_ratio(np.array(all_returns))

    # Add overall metrics to results
    results_df["overall_annualized_return"] = overall_annualized_return
    results_df["overall_sharpe_ratio"] = overall_sharpe_ratio
    results_df["overall_sortino_ratio"] = overall_sortino_ratio

    # Add constant parameters to the results
    results_df["max_cluster_labels"] = MAX_CLUSTER_LABELS
    results_df["num_clusters"] = NUM_CLUSTERS
    results_df["clustering_algorithm"] = CLUSTERING_ALGORITHM
    results_df["train_period"] = TRAIN_PERIOD
    results_df["test_period"] = TEST_PERIOD
    results_df["random_seed"] = RANDOM_SEED

    # Print results
    print(results_df)
    print("Backtesting completed.")


if __name__ == "__main__":
    main()
