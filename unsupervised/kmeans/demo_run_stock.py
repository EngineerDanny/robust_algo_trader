import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from numba import jit
import joblib
import os
import shutil
import json

sys.path.append(
    os.path.abspath(
        "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans"
    )
)
from utils import *


# Constants
# Define large value for cases with no losses (instead of infinity)
LARGE_VALUE = 1000.0 

# Load trading parameters from CSV
trading_params = pd.read_csv("params.csv")
param_row = 0 if len(sys.argv) != 2 else int(sys.argv[1])
param_dict = dict(trading_params.iloc[param_row, :]) 

# Extract trading parameters
INSTRUMENT = param_dict["instrument"]
PRICE_HISTORY_LENGTH = int(param_dict["price_history_length"])
NUM_PERCEPTUALLY_IMPORTANT_POINTS = int(param_dict["num_perceptually_important_points"])
NUM_CLUSTERS = int(param_dict["num_clusters"])
CLUSTERING_ALGORITHM = param_dict["clustering_algorithm"]
RANDOM_SEED = int(param_dict["random_seed"])
TEST_PERIOD = int(param_dict["test_period"])
TRAIN_PERIOD = int(param_dict["train_period"])
REVERSE_TEST = bool(param_dict["reverse_test"])

# Define clustering algorithms
clustering_estimator_dict = {
    "kmeans": KMeans(n_clusters=NUM_CLUSTERS, 
                     random_state=RANDOM_SEED,
                        n_init=1000),
    "gaussian_mixture": GaussianMixture(
        n_components=NUM_CLUSTERS, 
        covariance_type="tied", 
        random_state=RANDOM_SEED
    ),
    "birch": Birch(n_clusters=NUM_CLUSTERS)
}


def prepare_test_data(price_subset, max_trades_per_day=1):
    data_list = []
    scaler = StandardScaler()
    
    # Get instrument's high liquidity hours
    liquidity_start = 13
    liquidity_end = 20
    
    # Keep track of number of trades per day
    daily_trade_count = {}

    for index in range(PRICE_HISTORY_LENGTH, len(price_subset)):
        # Get price history for PIP calculation
        price_history = (
            price_subset["close"]
            .iloc[max(0, index - PRICE_HISTORY_LENGTH) : index]
            .values
        )
        if len(price_history) < PRICE_HISTORY_LENGTH:
            break

        # Current row index
        j = index - 1
        current_time = price_subset.index[j]
        current_hour = current_time.hour
        current_date = current_time.date()
        
        # Skip if we already took maximum trades for this day
        if daily_trade_count.get(current_date, 0) >= max_trades_per_day:
            continue
        
        # Check if current time is within high liquidity period
        # Handle cases where high liquidity period crosses midnight
        is_liquid_time = False
        if liquidity_start <= liquidity_end:
            is_liquid_time = liquidity_start <= current_hour < liquidity_end
        else:  # Period crosses midnight
            is_liquid_time = current_hour >= liquidity_start or current_hour < liquidity_end
            
        if not is_liquid_time:
            continue
            
        # Find current day's end time (15 mins before actual EOD)
        eod_time = pd.Timestamp.combine(
            current_time.date(), 
            pd.Timestamp('21:45').time()  # 15 mins before midnight
        ).tz_localize('UTC')

        # Get the EOD price (23:45 current day)
        eod_data = price_subset[price_subset.index <= eod_time]
        if len(eod_data) == 0:
            continue
        eod_row = eod_data.iloc[-1]
        
        # Calculate PIPs and scale them
        _, important_points = find_perceptually_important_points(
            price_history, NUM_PERCEPTUALLY_IMPORTANT_POINTS
        )
        scaled_points = scaler.fit_transform(important_points.reshape(-1, 1)).flatten()
        
        # Create data point with scaled PIPs
        data_point = {
            f"price_point_{i}": scaled_points[i]
            for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        }

        # Add time features
        data_point.update(
            price_subset.iloc[j][
                ["year", "month", "day_of_week", "hour", "minute"]
            ].to_dict()
        )
        
        # Calculate log return from current point to EOD
        data_point["trade_outcome"] = (eod_row["log_close"] - price_subset["log_close"].iloc[j])
        
        # Increment trade count for this day
        daily_trade_count[current_date] = daily_trade_count.get(current_date, 0) + 1
        
        data_list.append(data_point)

    return pd.DataFrame(data_list)


# @jit(nopython=True)
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

# @jit(nopython=True)
def calculate_point_distance(
    data, point_indices, point_prices, index, left_adj, right_adj
):
    time_diff = point_indices[right_adj] - point_indices[left_adj]
    price_diff = point_prices[right_adj] - point_prices[left_adj]
    slope = price_diff / time_diff
    x, y = index, data[index]
    return (
        (point_indices[left_adj] - x) ** 2 + (point_prices[left_adj] - y) ** 2
    ) ** 0.5 + (
        (point_indices[right_adj] - x) ** 2 + (point_prices[right_adj] - y) ** 2
    ) ** 0.5


def prepare_data(price_subset):
    data_list = []
    scaler = StandardScaler()
    
    # Get instrument's high liquidity hours
    liquidity_start = 13
    liquidity_end = 20

    for index in range(PRICE_HISTORY_LENGTH, len(price_subset)):
        # Get price history for PIP calculation
        price_history = (
            price_subset["close"]
            .iloc[max(0, index - PRICE_HISTORY_LENGTH) : index]
            .values
        )
        if len(price_history) < PRICE_HISTORY_LENGTH:
            break

        # Current row index
        j = index - 1
        current_time = price_subset.index[j]
        current_hour = current_time.hour
        
        # Check if current time is within high liquidity period
        # Handle cases where high liquidity period crosses midnight
        is_liquid_time = False
        if liquidity_start <= liquidity_end:
            is_liquid_time = liquidity_start <= current_hour < liquidity_end
        else:  # Period crosses midnight
            is_liquid_time = current_hour >= liquidity_start or current_hour < liquidity_end
            
        if not is_liquid_time:
            continue
            
        # Find current day's end time (15 mins before actual EOD)
        eod_time = pd.Timestamp.combine(
            current_time.date(), 
            pd.Timestamp('21:45').time()  # 15 mins before midnight
        ).tz_localize('UTC')

        # Get the EOD price (23:45 current day)
        eod_data = price_subset[price_subset.index <= eod_time]
        if len(eod_data) == 0:
            continue
        eod_row = eod_data.iloc[-1]
        
        # Calculate PIPs and scale them
        _, important_points = find_perceptually_important_points(
            price_history, NUM_PERCEPTUALLY_IMPORTANT_POINTS
        )
        scaled_points = scaler.fit_transform(important_points.reshape(-1, 1)).flatten()
        
        # Create data point with scaled PIPs
        data_point = {
            f"price_point_{i}": scaled_points[i]
            for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)
        }

        # Add time features
        data_point.update(
            price_subset.iloc[j][
                ["year", "month", "day_of_week", "hour", "minute"]
            ].to_dict()
        )
        
        # Calculate log return from current point to EOD (next day 00:00)
        data_point["trade_outcome"] = (eod_row["log_close"] -  price_subset["log_close"].iloc[j])
        # print(f"Processing {current_time} to {eod_time}...")
        data_list.append(data_point)

    return pd.DataFrame(data_list)

def evaluate_cluster_performance_df(price_data_df, train_best_cluster, clustering_model):
    # Prepare features for prediction
    price_point_columns = [f"price_point_{i}" for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)]
    feature_columns = price_point_columns + ["day_of_week", "hour", "minute"]
    
    # Predict clusters for test data
    price_features = price_data_df[feature_columns].values
    
    # scale features to 2 decimal places
    price_features = np.round(price_features, 2)
    price_data_df["cluster_label"] = clustering_model.predict(price_features)
    
    # Get the best cluster label and its direction from training
    cluster_label = train_best_cluster['cluster_label']
    trade_direction = train_best_cluster['trade_direction']
    
    # Get trades for the best cluster
    cluster_data = price_data_df[price_data_df['cluster_label'] == cluster_label]
    
    # Get trade outcomes and adjust for direction
    cluster_trades = cluster_data['trade_outcome']
    
    if REVERSE_TEST:
        trade_direction = 'short' if trade_direction == 'long' else 'long'
    
    if trade_direction == 'short':
        cluster_trades = -cluster_trades
    
    # Basic performance metrics
    total_return = cluster_trades.sum()
    num_trades = len(cluster_trades)
    
    # Create performance dictionary
    cluster_performance = {
        "cluster_label": cluster_label,
        "trade_direction": trade_direction,
        "actual_return": total_return,
        "num_trades": num_trades
    }
    return cluster_performance

def cluster_and_evaluate_price_data(price_data_df):
    price_point_columns = [f"price_point_{i}" for i in range(NUM_PERCEPTUALLY_IMPORTANT_POINTS)]
    feature_columns = price_point_columns + ["day_of_week", "hour", "minute"]
    price_features = price_data_df[feature_columns].values
    
    # scale features to 2 decimal places
    price_features = np.round(price_features, 2)
    
    clustering_model = clustering_estimator_dict[CLUSTERING_ALGORITHM]
    clustering_model.fit(price_features)
    price_data_df["cluster_label"] = clustering_model.predict(price_features)

    cluster_metrics = []
    for cluster in price_data_df['cluster_label'].unique():
        cluster_data = price_data_df[price_data_df['cluster_label'] == cluster]
        
        if len(cluster_data) < 5:  # Skip clusters with too few trades
            continue
            
        # Determine if cluster is long or short based on mean return
        cluster_mean = cluster_data['trade_outcome'].mean()
        is_long_cluster = cluster_mean >= 0
        
        # Adjust trade outcomes for short trades
        cluster_trades = cluster_data['trade_outcome']
        if not is_long_cluster:
            cluster_trades = -cluster_trades  # Invert returns for short trades
            
        # Basic metrics
        wins = cluster_trades > 0
        losses = cluster_trades < 0
        
        # 1. Win Rate
        win_rate = np.mean(wins) if len(cluster_trades) > 0 else 0
        
        # 2. Risk-adjusted return
        # Consider dividing by another factor to normalize
        returns_mean = cluster_trades.mean()
        returns_std = cluster_trades.std()
        annualization_factor = np.sqrt(252)
        actual_returns_mean = (np.exp(returns_mean) - 1)
        actual_returns_std = returns_std
        # Add scaling factor for 15-min returns
        scale_factor = 0.5  # This can be adjusted
        sharpe = (actual_returns_mean / actual_returns_std) * annualization_factor * scale_factor if actual_returns_std != 0 else 0
                
        # 3. Maximum Drawdown
        cumulative = cluster_trades.cumsum()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative - running_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else LARGE_VALUE
        
        # 4. Profit Factor
        gross_profits = cluster_trades[wins].sum() if any(wins) else 0
        gross_losses = abs(cluster_trades[losses].sum()) if any(losses) else 0
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else LARGE_VALUE
        
        # 5. Win/Loss Ratio
        avg_win = cluster_trades[wins].mean() if any(wins) else 0
        avg_loss = abs(cluster_trades[losses].mean()) if any(losses) else LARGE_VALUE
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else LARGE_VALUE
        
        # 6. Consistency Score (lower is better)
        returns_volatility = cluster_trades.std()
        
        cluster_metrics.append({
            'cluster_label': cluster,
            'trade_direction': 'long' if is_long_cluster else 'short',
            'num_trades': len(cluster_trades),
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'returns_volatility': returns_volatility,
            'avg_return': returns_mean,
            'actual_return': cluster_trades.sum(),
            'mean_return': cluster_mean
        })
    metrics_df = pd.DataFrame(cluster_metrics)
    
    print(f"Found {sharpe} sharpe.")
        
#     # Filter for good clusters with realistic performance metrics
#     mask = (
#         (metrics_df['win_rate'].between(0.5, 0.65)) &     # 55-65% win rate
#         (metrics_df['profit_factor'].between(1.5, 2.0)) &  # 1.6-2.0 profit factor
#         (metrics_df['win_loss_ratio'].between(1.2, 2.0)) &  # 1.2-2.0 win/loss ratio
#         (metrics_df['sharpe'].between(0.5, 2.0)) 
#     )
#     # Create an explicit copy of the filtered DataFrame
#     good_clusters = metrics_df[mask].copy()   
#     # Return empty DataFrame if no valid clusters
#     if len(good_clusters) == 0:
#        return {}, clustering_model 
#    # Calculate composite score for filtered clusters
#     good_clusters['consistency_score'] = (
#         good_clusters['win_rate'] / 0.65 * 0.3 +            # Normalized to max 0.65, 30% weight
#         good_clusters['profit_factor'] / 2.0 * 0.2 +        # Normalized to max 2.0, 30% weight
#         good_clusters['win_loss_ratio'] / 2.0 * 0.2 +       # Normalized to max 2.0, 20% weight
#         good_clusters['sharpe'] / 3.0 * 0.2                 # Normalized to max 2.0, 20% weight
#     )
#     print(f"Found {len(good_clusters)} good clusters.")

    mask = (
        # (metrics_df['sharpe'] > 0)          # Win rate above 55%
        (metrics_df['profit_factor'] > 1.5) 
        # (metrics_df['sharpe'] <= 5)             # Sharpe above 0.5
        # (metrics_df['win_loss_ratio'] > 1.2)    # Win/loss ratio above 1.2
    )
    # Create an explicit copy of the filtered DataFrame
    good_clusters = metrics_df[mask].copy()
    # Return empty DataFrame if no valid clusters
    if len(good_clusters) == 0:
        return {}, clustering_model 
    # Calculate composite score for filtered clusters
    good_clusters['consistency_score'] = (
         good_clusters['sharpe']
        # good_clusters['win_rate'] / 0.8 * 0.5 +          # Normalized to max 0.8, 30% weight
        # good_clusters['profit_factor'] / 2.0 * 0.5      # Normalized to max 2.0, 20% weight
        # good_clusters['win_loss_ratio'] / 2.0 * 0.2     # Normalized to max 2.0, 20% weight
        # good_clusters['sharpe'] / 3.0 * 0.3              # Normalized to max 3.0, 30% weight
    )
    print(f"Found {len(good_clusters)} good clusters.")
            
    # Get single best cluster by consistency score
    best_cluster_df = (
        good_clusters
        .sort_values('consistency_score', ascending=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_cluster_dict = best_cluster_df.iloc[0].to_dict()
    return best_cluster_dict, clustering_model


PROJECT_DIR = "/projects/genomic-ml/da2343/ml_project_2"
# PROJECT_DIR = "/Users/newuser/Projects/robust_algo_trader"
# Load the config file
config_path = f"{PROJECT_DIR}/settings/config_gfd.json"
with open(config_path) as f:
    config = json.load(f)

# instrument_dict = config["traded_instruments"][INSTRUMENT.split("_M15")[0]]
time_scaler = joblib.load(f"{PROJECT_DIR}/unsupervised/kmeans/ts_scaler_2018.joblib")
price_data = pd.read_csv(
    f"{PROJECT_DIR}/data/gen_alpaca_data/{INSTRUMENT}_raw_data.csv",
    parse_dates=["time"],
    index_col="time",
)

# Filter date range and apply time scaling
price_data = price_data.loc["2016-01-01":"2024-06-01"]
price_data['log_close'] = np.log(price_data['close'])
price_data['log_open'] = np.log(price_data['open'])
price_data["year"] = price_data.index.year
price_data["month"] = price_data.index.month
price_data["day_of_week"] = price_data.index.dayofweek
price_data["hour"] = price_data.index.hour
price_data["minute"] = price_data.index.minute
time_columns = ["day_of_week", "hour", "minute"]
price_data[time_columns] = time_scaler.transform(price_data[time_columns])
# Round price columns
columns_to_round = ['open', 'high', 'low', 'close', 'log_close', "log_open", "day_of_week", "hour", "minute"]
price_data[columns_to_round] = price_data[columns_to_round].round(6)

# Initialize the sliding window splitter for backtesting
window_splitter = OrderedSlidingWindowSplitter(
    train_weeks=TRAIN_PERIOD, test_weeks=TEST_PERIOD, step_size=1
)

backtest_results = []
for window, (train_indices, test_indices) in enumerate(window_splitter.split(price_data), 1):
    # if window <= 300:
    #     continue
    
    print(f"Processing window {window}...")
    train_data = price_data.iloc[train_indices, :]
    test_data = price_data.iloc[test_indices, :]

    # Prepare training data and perform clustering
    print("Preparing training data and clustering...")
    train_price_data = prepare_data(train_data)
    train_cluster_perf, clustering_model = cluster_and_evaluate_price_data(train_price_data)
    if not train_cluster_perf:
        continue

    # Prepare test data and evaluate cluster performance
    test_price_data = prepare_test_data(test_data)
    print("Preparing test data and evaluating cluster performance...")
    test_cluster_perf = evaluate_cluster_performance_df(test_price_data, train_cluster_perf, clustering_model)
    
    # check if test_cluster_perf dict is empty
    if not test_cluster_perf:
        continue

    # Compile results for this window
    print("Compiling results...")
    window_result = {
        "window": window,
        # Training metrics (single best cluster)
        "train_actual_return": train_cluster_perf["actual_return"], 
        "train_num_trades": train_cluster_perf["num_trades"], 
        "train_direction": train_cluster_perf["trade_direction"],
        
        # Test metrics (single cluster performance)
        "test_actual_return": test_cluster_perf["actual_return"],
        "test_num_trades": test_cluster_perf["num_trades"], 
        "test_direction": test_cluster_perf["trade_direction"]
    }
    backtest_results.append(window_result)
    if window > 300:
    # if window > 400:
        break
    

# Create base DataFrame from backtest results
backtest_results_df = pd.DataFrame(backtest_results)

# Get returns series
# Calculate metrics for train returns
train_returns = backtest_results_df['train_actual_return']
train_sharpe = (
    np.mean(train_returns) / np.std(train_returns)
    if np.std(train_returns) != 0 else LARGE_VALUE
)
train_winning_trades = train_returns[train_returns > 0]
train_losing_trades = train_returns[train_returns < 0]
train_gross_profits = train_winning_trades.sum() if len(train_winning_trades) > 0 else 0
train_gross_losses = abs(train_losing_trades.sum()) if len(train_losing_trades) > 0 else 0
train_profit_factor = train_gross_profits / train_gross_losses if train_gross_losses != 0 else LARGE_VALUE

# Calculate metrics for test returns
test_returns = backtest_results_df['test_actual_return']
test_sharpe = (
    np.mean(test_returns) / np.std(test_returns)
    if np.std(test_returns) != 0 else LARGE_VALUE
)
test_winning_trades = test_returns[test_returns > 0]
test_losing_trades = test_returns[test_returns < 0]
test_gross_profits = test_winning_trades.sum() if len(test_winning_trades) > 0 else 0
test_gross_losses = abs(test_losing_trades.sum()) if len(test_losing_trades) > 0 else 0
test_profit_factor = test_gross_profits / test_gross_losses if test_gross_losses != 0 else LARGE_VALUE

# Train Performance metrics
backtest_results_df['train_actual_return'] = round(backtest_results_df['train_actual_return'], 6)
backtest_results_df["train_average_return"] = round(train_returns.mean(), 6)
backtest_results_df["train_sharpe_ratio"] = round(train_sharpe, 6)
backtest_results_df["train_profit_factor"] = round(train_profit_factor, 6)
backtest_results_df["train_total_trades"] = backtest_results_df['train_num_trades'].sum()
backtest_results_df["train_avg_trades_per_window"] = round(backtest_results_df['train_num_trades'].mean(), 6)
backtest_results_df["train_win_ratio"] = round((train_returns > 0).mean() + (train_returns == 0).mean()/2, 6)
backtest_results_df["train_cum_return"] = round(train_returns.cumsum().values[-1], 6)

# Test Performance metrics
backtest_results_df['test_actual_return'] = round(backtest_results_df['test_actual_return'], 6)
backtest_results_df["test_average_return"] = round(test_returns.mean(), 6)
backtest_results_df["test_sharpe_ratio"] = round(test_sharpe, 6)
backtest_results_df["test_profit_factor"] = round(test_profit_factor, 6)
backtest_results_df["test_total_trades"] = backtest_results_df['test_num_trades'].sum()
backtest_results_df["test_avg_trades_per_window"] = round(backtest_results_df['test_num_trades'].mean(), 6)
backtest_results_df["test_win_ratio"] = round((test_returns > 0).mean() + (test_returns == 0).mean()/2, 6)
backtest_results_df["test_cum_return"] = round(test_returns.cumsum().values[-1], 6)

# General statistics
backtest_results_df["total_windows"] = len(backtest_results_df)
# Configuration parameters (no rounding needed for these)
backtest_results_df["reverse_test"] = REVERSE_TEST
backtest_results_df["num_clusters"] = NUM_CLUSTERS
backtest_results_df["clustering_algorithm"] = CLUSTERING_ALGORITHM
backtest_results_df["train_period"] = TRAIN_PERIOD
backtest_results_df["test_period"] = TEST_PERIOD
backtest_results_df["random_seed"] = RANDOM_SEED
backtest_results_df["instrument"] = INSTRUMENT
backtest_results_df["num_perceptually_important_points"] = NUM_PERCEPTUALLY_IMPORTANT_POINTS
backtest_results_df["price_history_length"] = PRICE_HISTORY_LENGTH


# save results to csv
out_file = f"results/{param_row}.csv"
backtest_results_df.to_csv(out_file, encoding="utf-8", index=False)
print("Backtesting completed.")