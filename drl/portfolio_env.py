import numpy as np
import pandas as pd
import datetime as dt
import os
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sklearn.preprocessing import MinMaxScaler
import torch as th
import sys
import os

sys.path.append("/Users/newuser/Projects/robust_algo_trader/drl/")
from ohlc_generator import SimpleOHLCGenerator 


# save dir should add the current date and time
SAVE_DIR = f"/Users/newuser/Projects/robust_algo_trader/drl/models/model_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(SAVE_DIR, exist_ok=True)

# DATA_DIR must be appended before the filename
# DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_synthetic_data/preprocessed_data"
DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_alpaca_data"

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 stock_data_list,
                 mode = "train",     
                 n_stocks = 10, 
                 episode_length = 12, # 12 months
                 temperature = 0.3, 
                 window_size = 252, # 1 year of data
                 episodes_per_dataset=50):
        
        super(PortfolioEnv, self).__init__()

        self.stock_data_list = stock_data_list
        self.mode = mode
        self.n_stocks = n_stocks
        self.episode_length = episode_length
        self.temperature = temperature
        self.window_size = window_size
        self.episodes_per_dataset = episodes_per_dataset
        self.stocks = None 
        
        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
        assert stock_data_list is not None, "stock_data_list cannot be None"
        assert len(stock_data_list) >= n_stocks, \
            f"Not enough stocks provided. Required: {n_stocks}, provided: {len(stock_data_list)}"
        
        # For tracking training progression
        if self.mode == "train":
            self.training_stage = 1  # Start with pure synthetic
            self.episode_count = 0
            self.current_dataset_episodes = 0 
            self.stage_transitions = {
                1: 500,   # Move to stage 2 after 500 episodes
                2: 1500   # Move to stage 3 after 1500 episodes
            }

        # Use raw features instead of pre-scaled ones
        self.features = [
            'Close', 'MA5', 'MA20', 'MA50', 'MA200',
            'RSI', 'BB_width', 'ATR', 'Return_1W',
            'Return_1M', 'Return_3M', 'CurrentDrawdown',
            'MaxDrawdown_252d', 'Sharpe_20d', 'Sharpe_60d'
        ]

        obs_dim = len(self.features) * self.n_stocks
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(obs_dim,), 
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.n_stocks,),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.mode == "train":
            return self._train_reset(seed) 
        else:
            return self._test_reset(seed)


    def step(self, action):
        allocation = self._convert_to_allocation(action)
        self.previous_allocation = allocation.copy()
        portfolio_return, stock_returns = self._calculate_monthly_performance(allocation)
        self.portfolio_value *= (1 + portfolio_return)

        # Get raw metrics (not scaled) for reward calculation
        sharpe = self._calculate_portfolio_metric('Sharpe_20d', allocation)
        max_drawdown = self._calculate_portfolio_metric('MaxDrawdown_252d', allocation)
        reward = self._calculate_reward(portfolio_return, sharpe, max_drawdown)
        self.monthly_returns.append(portfolio_return)

        info = {
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'allocation': allocation.copy(),
            'stock_returns': stock_returns
        }

        self.current_step += 30
        self.current_month += 1

        terminated = (self.current_month >= self.episode_length)
        truncated = False
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info


    def _train_reset(self, seed=None):
        self.episode_count += 1
        if self.episode_count == self.stage_transitions.get(1):
            self.training_stage = 2
            print("Training stage 2: Using synthetic data based on real data properties")
        elif self.episode_count == self.stage_transitions.get(2):
            self.training_stage = 3 
            print("Training stage 3: Using real data for training")
        
        # Stage 1 & 2: Synthetic Data
        if self.training_stage < 3:
            if (self.stocks is None or self.current_dataset_episodes >= self.episodes_per_dataset):
                # Generate appropriate synthetic data based on stage
                if self.training_stage == 2:
                    generator = SimpleOHLCGenerator()
                    synthetic_stocks = generator.generate_synthetic_data(
                        n_stocks=self.n_stocks,
                        seed=seed
                    )
                else:  
                    df = self.stock_data_list[np.random.randint(len(self.stock_data_list))]
                    generator = SimpleOHLCGenerator(df)
                    synthetic_stocks = generator.generate_bootstrap_data(
                        num_samples=self.n_stocks, 
                        segment_length=5
                    )
                
                # Create stocks dictionary with synthetic stocks
                self.stocks = {
                    f"stock_{i}": df for i, df in enumerate(synthetic_stocks)
                }
                self.current_dataset_episodes = 0
            self.current_dataset_episodes += 1
        # Stage 3: Real Data
        else:
            generator = SimpleOHLCGenerator()
            # Randomly select stocks
            selected_indices = np.random.choice(
                len(self.stock_data_list), 
                self.n_stocks, 
                replace=False
            )
            # Process real data - add technical indicators
            processed_stocks = {
                f"stock_{i}": generator.add_technical_indicators(self.stock_data_list[idx])
                for i, idx in enumerate(selected_indices)
            }
            
            # Find minimum length and align all stocks
            min_length = min(len(df) for df in processed_stocks.values())
            aligned_length = (min_length // 30) * 30
            
            # Align all processed stocks
            self.stocks = {
                stock_name: df.iloc[-aligned_length:].reset_index(drop=True)
                for stock_name, df in processed_stocks.items()
            }
            print(f"Training with real data: aligned to {aligned_length} data points")
        
        # Determine safe bounds for episode
        data_length = min(len(df) for df in self.stocks.values())
        max_start_idx = max(0, data_length - self.episode_length * 30 - 20)
        
        # Random start point (safely within bounds)
        self.current_step = np.random.randint(0, max_start_idx)
        self.current_month = 0
        self.monthly_returns = []
        self.portfolio_value = 100.0
        self.previous_allocation = np.zeros(self.n_stocks)
        
        observation = self._get_observation()
        info = {
            "mode": "train",
            "training_stage": self.training_stage,
            "episode_count": self.episode_count
        }
        return observation, info

    def _test_reset(self, seed=None):
        selected_indices = list(range(self.n_stocks))
        generator = SimpleOHLCGenerator()
        # First add technical indicators to all stocks
        processed_stocks = {
            f"stock_{i}": generator.add_technical_indicators(self.stock_data_list[idx])
            for i, idx in enumerate(selected_indices)
        }
        
        # Find minimum length after adding indicators
        min_length = min(len(df) for df in processed_stocks.values())
        aligned_length = (min_length // 30) * 30
        if aligned_length < self.episode_length * 30:
            raise ValueError(f"Not enough data for a full episode after adding indicators. " 
                            f"Need at least {self.episode_length * 30} points, but only have {aligned_length}.")
        
        # Align all processed stocks
        self.stocks = {
            stock_name: df.iloc[-aligned_length:].reset_index(drop=True)
            for stock_name, df in processed_stocks.items()
        }
        
        print(f"Testing with stocks: {selected_indices}")
        print(f"Aligned all test stocks to length {aligned_length} (multiple of 30)")
        
        # For testing, always start from the beginning of the time series
        self.current_step = 0
        self.current_month = 0
        self.monthly_returns = []
        self.portfolio_value = 100.0
        self.previous_allocation = np.zeros(self.n_stocks)
        
        observation = self._get_observation()
        info = {
            "mode": "test",
            "selected_stocks": selected_indices,
            "data_length": aligned_length,
            "max_episodes": aligned_length // (30 * self.episode_length)
        }
        return observation, info
    
    def _get_observation(self):
        observation = []
        for stock_name, stock_data in self.stocks.items():
            # Get window for scaling (including current step)
            window_start = max(0, self.current_step - self.window_size + 1) 
            window_end = self.current_step + 1 
            window_data = stock_data.iloc[window_start:window_end]

            # Scale all features for the entire window
            scaled_features = []
            for feature in self.features:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                feature_values = window_data[feature].values.reshape(-1, 1)
                scaled_window = scaler.fit_transform(feature_values)
                # Get the scaled value for the current step (last value in the window)
                scaled_val = scaled_window[-1][0]  # Last row, first column
                scaled_features.append(scaled_val)
            observation.extend(scaled_features)
        return np.array(observation, dtype=np.float32)

    def _convert_to_allocation(self, action_weights):
        raw_allocation = softmax(np.array(action_weights) / self.temperature)
        percentages = raw_allocation * 100
        # Apply discretization constraint (0%, 10%, 20%, 30%)
        # First, find the nearest valid allocation (multiples of 10%)
        allocations = np.round(percentages / 10) * 10
        allocations = np.clip(allocations, 0, 30)

        # Determine adjustment needed to sum to 100%
        total_allocation = np.sum(allocations)
        adjustment_needed = 100 - total_allocation

        if adjustment_needed != 0:
            # Calculate how close each stock was to the next discretization level
            distance_to_next = np.zeros_like(allocations)
            for i in range(len(allocations)):
                if adjustment_needed > 0 and allocations[i] < 30:
                    distance_to_next[i] = 10 - (percentages[i] % 10)
                elif adjustment_needed < 0 and allocations[i] > 0:
                    distance_to_next[i] = percentages[i] % 10
                else:
                    distance_to_next[i] = float('inf')

            # Prioritize adjustments for stocks closest to the next level
            num_adjustments = int(abs(adjustment_needed) // 10)
            if num_adjustments > 0:
                adjustment_indices = np.argsort(distance_to_next)[:num_adjustments]
                for idx in adjustment_indices:
                    if adjustment_needed > 0 and allocations[idx] < 30:
                        allocations[idx] += 10
                        adjustment_needed -= 10
                    elif adjustment_needed < 0 and allocations[idx] > 0:
                        allocations[idx] -= 10
                        adjustment_needed += 10
        return allocations

    def _calculate_monthly_performance(self, allocation):
        current_prices = np.array([
            self.stocks[f'stock_{i}'].iloc[self.current_step]['Close'] 
            for i in range(self.n_stocks)
        ])

        next_step = min(self.current_step + 30, len(next(iter(self.stocks.values()))) - 1)
        next_prices = np.array([
            self.stocks[f'stock_{i}'].iloc[next_step]['Close'] 
            for i in range(self.n_stocks)
        ])

        stock_returns = (next_prices - current_prices) / current_prices
        portfolio_return = np.sum((allocation / 100) * stock_returns)

        return portfolio_return, stock_returns

    def _calculate_portfolio_metric(self, metric_name, allocation):
        if not all(metric_name in stock_df.columns for stock_df in self.stocks.values()):
            return 0.0

        metric_values = np.array([
            self.stocks[f'stock_{i}'].iloc[self.current_step][metric_name] 
            for i in range(self.n_stocks)
        ])
        return np.sum((allocation / 100) * metric_values)

    def _calculate_reward(self, portfolio_return, sharpe, max_drawdown):
        # 1. IMMEDIATE REWARD COMPONENT (based on actual past performance)
        # Get the average return across all stocks as a benchmark
        benchmark_returns = np.mean([
            self.stocks[f'stock_{i}'].iloc[self.current_step].get('Return_1M', 0)
            for i in range(self.n_stocks)
        ])
        # Calculate excess return over benchmark
        excess_return = portfolio_return - max(0, benchmark_returns * 0.01)
        base_reward = excess_return * 100
        sharpe_component = sharpe * 1.0
        drawdown_component = max_drawdown * -1.5
        if max_drawdown < -0.1:
            drawdown_component *= 1.5
        immediate_reward = base_reward + sharpe_component + drawdown_component
        
        # 2. FUTURE REWARD COMPONENT (based on simulated future performance)
        weights = self.previous_allocation / 100.0
        _, stock_stats = self._simulate_future_price_paths()
        expected_future_return = 0
        future_var = 0
        for i in range(self.n_stocks):
            stock_name = f'stock_{i}'
            if weights[i] > 0:
                stock_stat = stock_stats[stock_name]
                expected_future_return += weights[i] * stock_stat['expected_returns'][-1]
                future_var += weights[i] * abs(stock_stat['final_var_95'])
        future_return_component = expected_future_return * 100
        future_risk_penalty = future_var * -100
        future_reward = future_return_component + future_risk_penalty
        total_reward = (0.7 * immediate_reward) + (0.3 * future_reward)
        return total_reward
    
    def _simulate_future_price_paths(self, n_simulations=50, horizon_months=3):
        # Store simulation results for each stock
        simulated_prices = {}
        stock_stats = {}
        
        # Use only recent data (last ~1 month of trading)
        recent_lookback = 21
        
        # Simulate future paths for each stock
        for i, stock_name in enumerate(self.stocks.keys()):
            stock_data = self.stocks[f'stock_{i}']
            current_step = self.current_step
            
            # Get current price
            current_price = stock_data.iloc[current_step]['Close']
            
            # Get only recent returns (what the agent can observe)
            start_idx = max(0, current_step - recent_lookback)
            recent_returns = stock_data['LogReturn'].iloc[start_idx:current_step+1].values
            
            # Calculate statistics from recent returns
            mean_return = np.mean(recent_returns)
            std_return = max(np.std(recent_returns), 1e-6)  # Prevent zero std
            
            # Initialize price paths for this stock
            stock_price_paths = np.zeros((n_simulations, horizon_months + 1))
            stock_price_paths[:, 0] = current_price  # Set initial price
            
            # Generate paths
            for sim in range(n_simulations):
                for month in range(1, horizon_months + 1):
                    # Sample a monthly return based on recent return distribution
                    # Scale daily mean and std to monthly
                    monthly_return = np.random.normal(mean_return * 21, std_return * np.sqrt(21))
                    
                    # Update price
                    stock_price_paths[sim, month] = stock_price_paths[sim, month-1] * np.exp(monthly_return)
            
            # Store price paths
            simulated_prices[stock_name] = stock_price_paths
            
            # Calculate statistics for this stock
            expected_prices = np.mean(stock_price_paths, axis=0)
            expected_returns = expected_prices / current_price - 1
            
            # Calculate risk metrics at final horizon
            final_prices = stock_price_paths[:, -1]
            final_returns = final_prices / current_price - 1
            var_95 = np.percentile(final_returns, 5)  # 5% worst case return
            expected_shortfall = np.mean(final_returns[final_returns < var_95])
            
            # Store stock statistics
            stock_stats[stock_name] = {
                'expected_prices': expected_prices,
                'expected_returns': expected_returns,
                'final_var_95': var_95,
                'expected_shortfall': expected_shortfall
            }
        
        return simulated_prices, stock_stats
    
    def render(self, mode='human'):
        print(f"Month {self.current_month}")
        print(f"Allocation: {self.previous_allocation}")
        if self.monthly_returns:
            print(f"Last month return: {self.monthly_returns[-1]:.4f}")
            print(f"Portfolio value: {self.portfolio_value:.2f}")

    def close(self):
        pass

#############################################################################
#############################################################################
## Helper functions to load data and evaluate the model
def get_stock_data_list(instrument_list):
    stock_data_list = []
    for instrument in instrument_list:
        file_path = f"{DATA_DIR}/{instrument}_D1_raw_data.csv"
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')
        stock_data_list.append(df)
        print(f"Loaded {instrument} with {len(df)} data points")
    return stock_data_list

def detailed_evaluation(trained_model, eval_env, n_episodes=10):
    all_allocations = []
    all_returns = []
    all_sharpes = []
    all_drawdowns = []
    monthly_allocations = []
    final_values = []
    
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_allocations = []
        episode_returns = []
        episode_sharpes = []
        episode_drawdowns = []
        done = False
        
        while not done:
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_allocations.append(info['allocation'])
            episode_returns.append(info['portfolio_return'])
            episode_sharpes.append(info['sharpe'])
            episode_drawdowns.append(info['max_drawdown'])
            
            if done:
                final_values.append(info['portfolio_value'])
        
        all_allocations.append(episode_allocations)
        monthly_allocations.extend(episode_allocations)
        all_returns.append(np.mean(episode_returns))
        all_sharpes.append(np.mean(episode_sharpes))
        all_drawdowns.append(np.mean(episode_drawdowns))
        
        print(f"Episode {episode+1}: Return = {np.mean(episode_returns):.4f}, Final Value = ${final_values[-1]:.2f}")
    
    avg_allocation = np.mean(monthly_allocations, axis=0)
    
    create_visualizations(
        avg_allocation, 
        all_returns, 
        all_sharpes, 
        all_drawdowns,
        final_values
    )
    
    return {
        'mean_return': np.mean(all_returns),
        'mean_sharpe': np.mean(all_sharpes),
        'mean_drawdown': np.mean(all_drawdowns),
        'mean_final_value': np.mean(final_values),
        'avg_allocation': avg_allocation
    }

def create_visualizations(avg_allocation, returns, sharpes, drawdowns, final_values):
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(avg_allocation)), avg_allocation)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', rotation=0)
        
    # create results dir if it doesn't exist
    RESULTS_DIR = f'{SAVE_DIR}/results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    
    plt.xlabel('Stock')
    plt.ylabel('Average Allocation (%)')
    plt.title('Average Portfolio Allocation')
    plt.xticks(range(len(avg_allocation)), [f'Stock {i}' for i in range(len(avg_allocation))])
    plt.ylim(0, max(avg_allocation) * 1.2)
    plt.savefig(f'{RESULTS_DIR}/portfolio_allocation.png')
    
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=10, alpha=0.7)
    plt.axvline(np.mean(returns), color='r', linestyle='dashed', linewidth=2)
    plt.text(np.mean(returns)*1.1, plt.ylim()[1]*0.9, f'Mean: {np.mean(returns):.4f}')
    plt.xlabel('Average Monthly Return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Monthly Returns')
    plt.savefig(f'{RESULTS_DIR}/returns_distribution.png')
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=10, alpha=0.7)
    plt.axvline(np.mean(final_values), color='r', linestyle='dashed', linewidth=2)
    plt.text(np.mean(final_values)*1.02, plt.ylim()[1]*0.9, f'Mean: ${np.mean(final_values):.2f}')
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Portfolio Values (12-month episodes)')
    plt.savefig(f'{RESULTS_DIR}/portfolio_values.png')
    
    plt.figure(figsize=(12, 6))
    metrics = ['Return (%)', 'Sharpe', 'Drawdown (%)']
    values = [np.mean(returns)*100, np.mean(sharpes), np.mean(drawdowns)*100]
    colors = ['green', 'blue', 'red']
    
    bars = plt.bar(metrics, values, color=colors)
    plt.title('Average Performance Metrics')
    
    for bar in bars:
        height = bar.get_height()
        sign = "+" if height > 0 else ""
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.6,
                 f'{sign}{height:.2f}',
                 ha='center', va='bottom' if height > 0 else 'top')
    
    plt.savefig(f'{SAVE_DIR}/results/performance_metrics.png')
    print("Visualizations saved to 'results' directory")

# Some helper functions to train and evaluate the model
def make_env(stock_data_list, rank=0, seed=0):
    def _init():
        env = Monitor(PortfolioEnv(
            stock_data_list, 
            mode="train"
        ))
        env.reset(seed=seed + rank)
        return env
    return _init    

def train_model(stock_data_list, total_timesteps=200_000):
    print("Creating environment...")
    n_envs = 8
    # Create multiple environments running in parallel
    # env = SubprocVecEnv([make_env(stock_data_list, i) for i in range(n_envs)])
    
    env = PortfolioEnv(stock_data_list)
    # check_env(env)
    
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy", 
        env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/portfolio_env_logs",
        verbose=1,
        device="mps",
        n_steps=256,
        learning_rate=1e-4,
        batch_size=128,
        # gamma=0.99,
        # ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256, 128, 128, 64],
            activation_fn=th.nn.Tanh,
        ),
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=SAVE_DIR,
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    final_model_path = os.path.join(SAVE_DIR, "ppo_portfolio_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    return model

def evaluate_model(stock_data_list, trained_model, n_episodes=10):
    print(f"Evaluating agent over {n_episodes} episodes...")
    eval_env = Monitor(PortfolioEnv(stock_data_list, 
                                    mode="test"))
    # eval_env = PortfolioEnv(stock_data_list)
    mean_reward, std_reward = evaluate_policy(
        trained_model, 
        eval_env, 
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    results = detailed_evaluation(trained_model, eval_env)
    
    print("\nPerformance Summary:")
    print(f"Average Monthly Return: {results['mean_return']:.4f}")
    print(f"Average Sharpe Ratio: {results['mean_sharpe']:.4f}")
    print(f"Average Max Drawdown: {results['mean_drawdown']:.4f}")
    print(f"Average Portfolio Allocation: {results['avg_allocation']}")
    print(f"Final Average Portfolio Value: ${results['mean_final_value']:.2f}")
    return results


if __name__ == "__main__":
    # TRAIN
    instrument_list = ["AAPL", "MSFT", "JNJ", "PG", "JPM", "NVDA", "AMD", "TSLA", "CRM", "AMZN"] 
    stock_data_list = get_stock_data_list(instrument_list)
    print("Training model...")
    trained_model = train_model(stock_data_list, total_timesteps=2_000)
    print("Training complete!")

    # EVALUATE
    eval_instrument_list = ["AAPL", "MSFT", "JNJ", "PG", "JPM", "NVDA", "AMD", "TSLA", "CRM", "AMZN"]        
    eval_stock_data_list = get_stock_data_list(eval_instrument_list)
    print("Evaluating model...")
    evaluate_model(eval_stock_data_list, trained_model)
    print("Evaluation complete!")