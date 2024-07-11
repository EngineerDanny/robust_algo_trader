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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define constants
CANDLES_PER_DAY = 4 * 24  # 15-minute candles
INITIAL_CAPITAL = 100000  # Starting capital for backtesting
RISK_FREE_RATE = 0.01  # Risk-free rate for Sharpe ratio calculation

def load_params(param_file="params.csv"):
    """Load trading parameters from a CSV file."""
    trading_params = pd.read_csv(param_file)
    param_row = 0 if len(sys.argv) != 2 else int(sys.argv[1])
    return dict(trading_params.iloc[param_row, :])

# Load parameters
param_dict = load_params()

# Extract trading parameters from the loaded dictionary
MAX_CLUSTER_LABELS = int(param_dict["max_cluster_labels"])
PRICE_HISTORY_LENGTH = int(param_dict["price_history_length"])
NUM_PERCEPTUALLY_IMPORTANT_POINTS = int(param_dict["num_perceptually_important_points"])
DISTANCE_MEASURE = int(param_dict["distance_measure"])
NUM_CLUSTERS = int(param_dict["num_clusters"])
ATR_MULTIPLIER = float(param_dict["atr_multiplier"])
CLUSTERING_ALGORITHM = param_dict["clustering_algorithm"]
RANDOM_SEED = int(param_dict["random_seed"])
TRAIN_PERIOD = int(param_dict["train_period"] * CANDLES_PER_DAY)
TEST_PERIOD = int(param_dict["test_period"] * CANDLES_PER_DAY)

# Define clustering algorithms
clustering_models = {
    "kmeans": KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED),
    "mini_batch_kmeans": MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED),
    "birch": Birch(n_clusters=NUM_CLUSTERS),
    "gaussian_mixture": GaussianMixture(n_components=NUM_CLUSTERS, covariance_type="tied", random_state=RANDOM_SEED),
}


def calculate_trading_metrics(trade_outcomes, initial_capital=INITIAL_CAPITAL, trading_days_per_year=252, risk_free_rate=RISK_FREE_RATE):
    """Calculate various trading performance metrics."""
    if len(trade_outcomes) == 0:
        return np.zeros(10)  # Return zeros if no trades

    # Calculate returns
    returns = trade_outcomes
    cumulative_return = np.sum(returns)
    
    # Annualize the return
    num_trading_days = len(trade_outcomes)
    annualized_return = ((1 + cumulative_return) ** (trading_days_per_year / num_trading_days)) - 1
    
    # Calculate drawdowns and max drawdown
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = peak - cumulative_returns
    max_drawdown = np.max(drawdowns)
    
    # Calculate Sharpe and Sortino ratios
    excess_returns = returns - (risk_free_rate / trading_days_per_year)
    sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(trading_days_per_year) if np.std(returns) != 0 else 0
    
    downside_returns = np.minimum(excess_returns, 0)
    sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(trading_days_per_year) if np.std(downside_returns) != 0 else 0
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
    
    # Calculate win rate and profit factors
    wins = trade_outcomes > 0
    losses = trade_outcomes < 0
    win_rate = np.sum(wins) / len(trade_outcomes)
    profit_factor = np.sum(trade_outcomes[wins]) / np.abs(np.sum(trade_outcomes[losses])) if np.sum(trade_outcomes[losses]) != 0 else 0
    
    # Calculate average trade metrics
    avg_trade = np.mean(trade_outcomes)

    return np.array([
        cumulative_return,
        annualized_return,
        max_drawdown,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        win_rate,
        profit_factor,
        avg_trade,
        len(trade_outcomes)
    ])

@jit(nopython=True)
def find_perceptually_important_points(price_data, num_points):
    """Find perceptually important points in a price series."""
    point_indices = np.zeros(num_points, dtype=np.int64)
    point_prices = np.zeros(num_points, dtype=np.float64)
    point_indices[0], point_indices[1] = 0, len(price_data) - 1
    point_prices[0], point_prices[1] = price_data[0], price_data[-1]

    for current_point in range(2, num_points):
        max_distance, max_distance_index, insert_index = 0.0, -1, -1
        for i in range(1, len(price_data) - 1):
            left_adj = np.searchsorted(point_indices[:current_point], i, side="right") - 1
            right_adj = left_adj + 1
            distance = calculate_point_distance(price_data, point_indices[:current_point], point_prices[:current_point], i, left_adj, right_adj)
            if distance > max_distance:
                max_distance, max_distance_index, insert_index = distance, i, right_adj

        point_indices[insert_index + 1 : current_point + 1] = point_indices[insert_index:current_point]
        point_prices[insert_index + 1 : current_point + 1] = point_prices[insert_index:current_point]
        point_indices[insert_index], point_prices[insert_index] = max_distance_index, price_data[max_distance_index]

    return point_indices, point_prices

@jit(nopython=True)
def calculate_point_distance(data, point_indices, point_prices, index, left_adj, right_adj):
    """Calculate the distance of a point from a line segment."""
    time_diff = point_indices[right_adj] - point_indices[left_adj]
    price_diff = point_prices[right_adj] - point_prices[left_adj]
    slope = price_diff / time_diff
    intercept = point_prices[left_adj] - point_indices[left_adj] * slope
    x, y = index, data[index]

    if DISTANCE_MEASURE == 1:
        return ((point_indices[left_adj] - x) ** 2 + (point_prices[left_adj] - y) ** 2) ** 0.5 + \
               ((point_indices[right_adj] - x) ** 2 + (point_prices[right_adj] - y) ** 2) ** 0.5
    elif DISTANCE_MEASURE == 2:
        return abs((slope * x + intercept) - y) / (slope**2 + 1) ** 0.5
    else:  # DISTANCE_MEASURE == 3
        return abs((slope * x + intercept) - y)

@jit(nopython=True)
def determine_trade_outcome(future_highs, future_lows, take_profit, stop_loss, current_price):
    """Determine the outcome of a trade based on future price movements."""
    tp_hit = np.argmax(future_highs >= take_profit)
    sl_hit = np.argmax(future_lows <= stop_loss)

    if tp_hit < sl_hit or (tp_hit > 0 and sl_hit == 0):
        return (take_profit - current_price) / current_price
    elif sl_hit < tp_hit or (sl_hit > 0 and tp_hit == 0):
        return (stop_loss - current_price) / current_price
    else:
        return 0

def prepare_price_data(price_data, history_length, num_pips):
    """Prepare price data for clustering by extracting features and calculating trade outcomes."""
    price_data_list = []
    scaler = StandardScaler()

    for index in range(history_length, len(price_data)):
        price_history = price_data["close"].iloc[index - history_length : index].values
        if len(price_history) < history_length:
            continue

        _, important_points = find_perceptually_important_points(price_history, num_pips)
        scaled_points = scaler.fit_transform(important_points.reshape(-1, 1)).flatten()

        data_point = {f"price_point_{i}": scaled_points[i] for i in range(num_pips)}
        data_point.update(price_data.iloc[index - 1][["year", "month", "day_of_week", "hour", "minute"]].to_dict())

        current_price = price_data["close"].iloc[index - 1]
        current_atr = price_data["atr"].iloc[index - 1]
        take_profit = current_price + (ATR_MULTIPLIER * current_atr)
        stop_loss = current_price - (ATR_MULTIPLIER * current_atr)

        future_highs = price_data["high"].iloc[index:].values
        future_lows = price_data["low"].iloc[index:].values

        data_point["trade_outcome"] = determine_trade_outcome(future_highs, future_lows, take_profit, stop_loss, current_price)
        price_data_list.append(data_point)

    return pd.DataFrame(price_data_list)

def cluster_and_evaluate_price_data(price_data_df):
    """Perform clustering on price data and evaluate the performance of each cluster."""
    price_features = price_data_df[[f"price_point_{i}" for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)] + 
                                   ["day_of_week", "hour", "minute"]].values
    clustering_model = clustering_models[CLUSTERING_ALGORITHM]
    clustering_model.fit(price_features)
    price_data_df["cluster_label"] = clustering_model.predict(price_features)

    top_clusters_df = price_data_df.groupby("cluster_label")["trade_outcome"].sum().abs().nlargest(MAX_CLUSTER_LABELS).reset_index()

    best_clusters_list = []
    for cluster_label in top_clusters_df["cluster_label"]:
        cluster_trade_outcomes = price_data_df[price_data_df["cluster_label"] == cluster_label]["trade_outcome"].values
        metrics = calculate_trading_metrics(cluster_trade_outcomes)
        best_clusters_list.append({
            "cluster_label": cluster_label,
            "total_return": metrics[0],
            "annualized_return": metrics[1],
            "max_drawdown": metrics[2],
            "sharpe_ratio": metrics[3],
            "sortino_ratio": metrics[4],
            "calmar_ratio": metrics[5],
            "win_rate": metrics[6],
            "profit_factor": metrics[7],
            "avg_trade": metrics[8],
            "num_trades": metrics[9]
        })

    return pd.DataFrame(best_clusters_list), clustering_model

def main():
    """Main function to run the trading strategy backtesting."""
    # Load time scaler
    time_scaler = joblib.load("ts_scaler_2018.joblib")

    # Load and preprocess price data
    price_data = pd.read_csv("/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/GBP_USD_M15_raw_data.csv",
                             parse_dates=["time"], index_col="time")

    # Extract time features
    price_data["year"] = price_data.index.year
    price_data["month"] = price_data.index.month
    price_data["day_of_week"] = price_data.index.dayofweek
    price_data["hour"] = price_data.index.hour
    price_data["minute"] = price_data.index.minute
    
    # Calculate ATR
    price_data["atr"] = talib.ATR(price_data["high"].values, price_data["low"].values, price_data["close"].values, timeperiod=14)

    # Filter date range and apply time scaling
    price_data = price_data.loc["2019-01-01":"2019-05-01"]
    time_columns = ["day_of_week", "hour", "minute"]
    price_data[time_columns] = np.round(time_scaler.transform(price_data[time_columns]), 6)
    price_data["atr"] = price_data["atr"].round(6)

    # Initialize the sliding window splitter for backtesting
    window_splitter = SlidingWindowSplitter(window_length=TRAIN_PERIOD, fh=np.arange(1, TEST_PERIOD + 1), step_length=TEST_PERIOD)

    backtest_results = []
    for window, (train_indices, test_indices) in enumerate(window_splitter.split(price_data)):
        print(f"Processing window {window}...")
        train_data = price_data.iloc[train_indices, :]
        test_data = price_data.iloc[test_indices, :]

        # Prepare training data and perform clustering
        print("Preparing training data and clustering...")
        train_price_data = prepare_price_data(train_data, PRICE_HISTORY_LENGTH, NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        train_best_clusters, clustering_model = cluster_and_evaluate_price_data(train_price_data)
        if train_best_clusters.empty:
            continue

        # Prepare test data and evaluate cluster performance
        print("Preparing test data and evaluating cluster performance...")
        test_price_data = prepare_price_data(test_data, PRICE_HISTORY_LENGTH, NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        test_cluster_performance = cluster_and_evaluate_price_data(test_price_data)[0]
        if test_cluster_performance.empty:
            continue

        # Compile results for this window
        print("Compiling results...")
        window_result = {
            "window": window,
            "train_total_return": np.sum(train_best_clusters["total_return"]),
            "train_annualized_return": (1 + np.sum(train_best_clusters["total_return"])) ** (252 / TRAIN_PERIOD) - 1,
            "train_sharpe_ratio": np.mean(train_best_clusters["sharpe_ratio"]),
            "train_sortino_ratio": np.mean(train_best_clusters["sortino_ratio"]),
            "train_calmar_ratio": np.mean(train_best_clusters["calmar_ratio"]),
            "train_win_rate": np.mean(train_best_clusters["win_rate"]),
            "train_profit_factor": np.mean(train_best_clusters["profit_factor"]),
            "train_num_trades": np.sum(train_best_clusters["num_trades"]),
            "test_total_return": np.sum(test_cluster_performance["total_return"]),
            "test_annualized_return": (1 + np.sum(test_cluster_performance["total_return"])) ** (252 / TEST_PERIOD) - 1,
            "test_sharpe_ratio": np.mean(test_cluster_performance["sharpe_ratio"]),
            "test_sortino_ratio": np.mean(test_cluster_performance["sortino_ratio"]),
            "test_calmar_ratio": np.mean(test_cluster_performance["calmar_ratio"]),
            "test_win_rate": np.mean(test_cluster_performance["win_rate"]),
            "test_profit_factor": np.mean(test_cluster_performance["profit_factor"]),
            "test_num_trades": np.sum(test_cluster_performance["num_trades"]),
        }
        backtest_results.append(window_result)

    # Compile final results
    results_df = pd.DataFrame(backtest_results)
    
    # Calculate cumulative metrics
    results_df["train_cumulative_return"] = (1 + results_df["train_total_return"]).cumprod() - 1
    results_df["test_cumulative_return"] = (1 + results_df["test_total_return"]).cumprod() - 1

    train_returns = results_df["train_total_return"].values
    test_returns = results_df["test_total_return"].values

    # Calculate overall Sharpe ratios
    results_df["train_overall_sharpe"] = (np.mean(train_returns) / np.std(train_returns)) * np.sqrt(252) if np.std(train_returns) != 0 else 0
    results_df["test_overall_sharpe"] = (np.mean(test_returns) / np.std(test_returns)) * np.sqrt(252) if np.std(test_returns) != 0 else 0

    # Add constant parameters to the results
    results_df["max_cluster_labels"] = MAX_CLUSTER_LABELS
    results_df["num_clusters"] = NUM_CLUSTERS
    results_df["clustering_algorithm"] = CLUSTERING_ALGORITHM
    results_df["train_period"] = TRAIN_PERIOD
    results_df["test_period"] = TEST_PERIOD
    results_df["random_seed"] = RANDOM_SEED

    # Print results
    print("\nBacktesting Results:")
    print(results_df)

    # Print overall performance metrics
    print("\nOverall Performance:")
    print(f"Train Cumulative Return: {results_df['train_cumulative_return'].iloc[-1]:.2%}")
    print(f"Test Cumulative Return: {results_df['test_cumulative_return'].iloc[-1]:.2%}")
    print(f"Train Overall Sharpe Ratio: {results_df['train_overall_sharpe'].iloc[-1]:.2f}")
    print(f"Test Overall Sharpe Ratio: {results_df['test_overall_sharpe'].iloc[-1]:.2f}")

    print("Backtesting completed.")

if __name__ == "__main__":
    main()