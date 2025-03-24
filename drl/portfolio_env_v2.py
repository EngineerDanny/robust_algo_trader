import numpy as np
import pandas as pd
import datetime as dt
import os
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sklearn.preprocessing import MinMaxScaler
import torch as th
import sys
import tensorflow as tf
import random


sys.path.append("/Users/newuser/Projects/robust_algo_trader/drl/")
from ohlc_generator import SimpleOHLCGenerator 


# save dir should add the current date and time
SAVE_DIR = f"/Users/newuser/Projects/robust_algo_trader/drl/models/model_{dt.datetime.now().strftime('%Y%m%d_%H')}"
os.makedirs(SAVE_DIR, exist_ok=True)

# DATA_DIR must be appended before the filename
# DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_synthetic_data/preprocessed_data"
DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_alpaca_data"
MASTER_SEED = 42

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 stock_data_list,
                 mode = "train",     
                 n_stocks = 10, 
                 episode_length = 12, # 12 months
                 temperature = 0.3, 
                 window_size = 252, # 1 year of data
                 episodes_per_dataset=500,
                 days_per_step=20,
                 total_timesteps=200_000,  # Default to the value in your train_model function
                 stage_transition_percentages=(1.0, 1.0),  # Percentages for stage transitions
                 n_parallel_envs=8,
                 seed = None
                ):
        
        super(PortfolioEnv, self).__init__()

        self.stock_data_list = stock_data_list
        self.mode = mode
        self.n_stocks = n_stocks
        self.episode_length = episode_length
        self.temperature = temperature
        self.window_size = window_size
        self.episodes_per_dataset = episodes_per_dataset
        self.days_per_step = days_per_step
        self.stocks = None 
        self.episode_count = 0
        
        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
        assert stock_data_list is not None, "stock_data_list cannot be None"
        assert len(stock_data_list) >= n_stocks, \
            f"Not enough stocks provided. Required: {n_stocks}, provided: {len(stock_data_list)}"
        
        # For tracking training progression
        if self.mode == "train":
            self.training_stage = 1  # Start with pure synthetic
            self.current_dataset_episodes = 0 
            expected_episodes = (total_timesteps / self.episode_length) // n_parallel_envs
            
            self.stage_transitions = {
                1: int(expected_episodes * stage_transition_percentages[0]),  # Move to stage 2
                2: int(expected_episodes * stage_transition_percentages[1])   # Move to stage 3
            }
        
        # Use raw features instead of pre-scaled ones
        self.features = [
            'Close', 'MA5', 'MA20', 'MA50', 'MA200',
            'RSI', 'BB_width', 'ATR', 'Return_1W',
            'Return_1M', 'Return_3M', 'CurrentDrawdown',
            'MaxDrawdown_252d', 'Sharpe_20d', 'Sharpe_60d'
        ]
        
        # Add simulation features
        self.simulation_features = [
            'Sim_Return_1M', 'Sim_Return_3M',
            'Sim_VaR_95', 'Sim_ExpShortfall', 
            'Sim_Volatility'
        ]

        # Update observation space dimension
        obs_dim = (len(self.features) + len(self.simulation_features)) * self.n_stocks
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
        # self.reset()
        self.seed_val = seed
        self.np_random = np.random.RandomState(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
        
        episode_seed = None
        
        if self.seed_val is not None:
            episode_seed = self.seed_val + self.episode_count
            np.random.seed(episode_seed)
            random.seed(episode_seed)

        if self.mode == "train":
            return self._train_reset(episode_seed) 
        else:
            return self._test_reset(episode_seed)


    def step(self, action):
        allocation = self._convert_to_allocation(action)
        self.previous_allocation = allocation.copy()
        portfolio_return, stock_returns = self._calculate_monthly_performance(allocation)
        self.portfolio_value *= (1 + portfolio_return)

        # Get raw metrics (not scaled) for reward calculation
        sharpe = self._calculate_portfolio_metric('Sharpe_20d', allocation)
        max_drawdown = self._calculate_portfolio_metric('MaxDrawdown_252d', allocation)
        reward = self._calculate_reward(portfolio_return, sharpe, max_drawdown, stock_returns)
        self.monthly_returns.append(portfolio_return)

        info = {
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'allocation': allocation.copy(),
            'stock_returns': stock_returns
        }

        self.current_step += self.days_per_step
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
            selected_indices = np.random.choice(
                len(self.stock_data_list), 
                self.n_stocks, 
                replace=False
            )
            processed_stocks = {
                f"stock_{i}": generator.add_technical_indicators(self.stock_data_list[idx])
                for i, idx in enumerate(selected_indices)
            }
            
            # Find minimum length and align all stocks
            min_length = min(len(df) for df in processed_stocks.values())
            aligned_length = (min_length // self.days_per_step) * self.days_per_step
            
            # Align all processed stocks
            self.stocks = {
                stock_name: df.iloc[-aligned_length:].reset_index(drop=True)
                for stock_name, df in processed_stocks.items()
            }
            print(f"Training with real data: aligned to {aligned_length} data points")
        
        # Determine safe bounds for episode
        data_length = min(len(df) for df in self.stocks.values())
        max_start_idx = max(0, data_length - self.episode_length * self.days_per_step - self.days_per_step) 
        
        # Random start point (safely within bounds)
        # self.current_step = np.random.randint(0, max_start_idx)
        self.current_step = 0
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
        self.episode_count += 1
        selected_indices = list(range(self.n_stocks))
        generator = SimpleOHLCGenerator()
        # First add technical indicators to all stocks
        processed_stocks = {
            f"stock_{i}": generator.add_technical_indicators(self.stock_data_list[idx])
            for i, idx in enumerate(selected_indices)
        }
        
        # Find minimum length after adding indicators
        min_length = min(len(df) for df in processed_stocks.values())
        aligned_length = (min_length // self.days_per_step) * self.days_per_step 
        if aligned_length < self.episode_length * self.days_per_step:
            raise ValueError(f"Not enough data for a full episode after adding indicators. " 
                f"Need at least {self.episode_length * self.days_per_step} points, but only have {aligned_length}.")
        
        # Align all processed stocks
        self.stocks = {
            stock_name: df.iloc[-aligned_length:].reset_index(drop=True)
            for stock_name, df in processed_stocks.items()
        }
        
        print(f"Testing with stocks: {selected_indices}")
        print(f"Aligned all test stocks to length {aligned_length} (multiple of {self.days_per_step})")  # Changed from 30
        
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
            "max_episodes": aligned_length // (self.days_per_step * self.episode_length)
        }
        return observation, info
    
    
    def _get_observation(self):
        _, stock_stats = self._simulate_future_price_paths(n_simulations=100)
        observation = []
        for i, (stock_name, stock_data) in enumerate(self.stocks.items()):
            # Get window for scaling (including current step)
            window_start = max(0, self.current_step - self.window_size + 1) 
            window_end = self.current_step + 1 
            window_data = stock_data.iloc[window_start:window_end]

            # Scale historical features (unchanged)
            scaled_features = []
            for feature in self.features:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                feature_values = window_data[feature].values.reshape(-1, 1)
                scaled_window = scaler.fit_transform(feature_values)
                scaled_val = scaled_window[-1][0]
                scaled_features.append(scaled_val)
            
            # Add simulation features (new)
            stock_stat = stock_stats[stock_name]
            
            # 1-month expected return
            scaled_features.append(np.clip(stock_stat['expected_returns'][0] * 5, -1, 1))
            
            # 3-month expected return
            scaled_features.append(np.clip(stock_stat['expected_returns'][-1] * 3, -1, 1))
            
            # VaR 95
            scaled_features.append(np.clip(stock_stat['final_var_95'] * 3, -1, 1))
            
            # Expected shortfall
            scaled_features.append(np.clip(stock_stat['expected_shortfall'] * 3, -1, 1))
            
            # Path volatility (new calculation)
            path_volatility = np.std(stock_stat['final_returns']) if 'final_returns' in stock_stat else 0.05
            scaled_features.append(np.clip(path_volatility * 10, -1, 1))
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
        next_step = min(self.current_step + self.days_per_step, len(next(iter(self.stocks.values()))) - 1)
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
    
    # def _calculate_reward(self, portfolio_return, sharpe, max_drawdown, stock_returns):
    #     # Calculate benchmark using equal weights over the SAME time period
    #     equal_weights = np.ones(self.n_stocks) / self.n_stocks
    #     benchmark_return = np.sum(equal_weights * stock_returns)
        
    #     # Simple excess return calculation
    #     excess_return = portfolio_return - benchmark_return
        
    #     # Scale for easier learning
    #     reward = excess_return * 100
        
    #     return reward
    
    
    def _calculate_reward(self, portfolio_return, sharpe, max_drawdown, stock_returns):
        # Calculate benchmark using equal weights over the SAME time period
        equal_weights = np.ones(self.n_stocks) / self.n_stocks
        benchmark_return = np.sum(equal_weights * stock_returns)
        
        # Calculate excess return
        excess_return = portfolio_return - benchmark_return
        
        # Simple reward history for smoothing
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        
        # Calculate current reward
        current_reward = excess_return * 100
        
        # Add to history
        self.reward_history.append(current_reward)
        
        # Keep only recent history (last 3 steps)
        if len(self.reward_history) > 3:
            self.reward_history = self.reward_history[-3:]
        
        # Return smoothed reward
        return np.mean(self.reward_history)
        
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
            current_price = stock_data.iloc[current_step]['Close']
            start_idx = max(0, current_step - recent_lookback)
            
            # Check if LogReturn exists in the data
            if 'LogReturn' not in stock_data.columns:
                # Calculate log returns from price data
                prices = stock_data['Close'].values
                if len(prices) > 1:
                    # Calculate log returns: ln(P_t / P_{t-1})
                    log_prices = np.log(prices)
                    log_returns = np.diff(log_prices)
                    
                    # Get recent returns based on current position
                    if current_step > 0:
                        idx_end = min(current_step, len(log_returns))
                        idx_start = max(0, idx_end - recent_lookback)
                        recent_returns = log_returns[idx_start:idx_end]
                    else:
                        # Fallback if at the start of the series
                        recent_returns = np.array([0.0001])  # Small positive default
                else:
                    # Fallback if insufficient price history
                    recent_returns = np.array([0.0001])
            else:
                # Use existing LogReturn column
                recent_returns = stock_data['LogReturn'].iloc[start_idx:current_step+1].values
                if len(recent_returns) == 0:
                    recent_returns = np.array([0.0001])  # Fallback for empty returns
            
            # Calculate return statistics
            mean_return = np.mean(recent_returns)
            std_return = max(np.std(recent_returns), 1e-6)  # Prevent zero std
            
            # Initialize price paths
            stock_price_paths = np.zeros((n_simulations, horizon_months + 1))
            stock_price_paths[:, 0] = current_price  # Set initial price
            
            # Generate paths using geometric Brownian motion
            for sim in range(n_simulations):
                for month in range(1, horizon_months + 1):
                    # Scale daily return to monthly (approx 21 trading days)
                    monthly_return = np.random.normal(mean_return * 21, std_return * np.sqrt(21))
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
                'expected_shortfall': expected_shortfall,
                'final_returns': final_returns 
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

# Some helper functions to train and evaluate the model
def make_env(stock_data_list, total_timesteps = 200_000, rank=0, seed=0):
    def _init():
        env = Monitor(PortfolioEnv(
            stock_data_list, 
            mode="train",
            total_timesteps=total_timesteps,
            seed= seed + rank, 
        ))
        return env
    return _init    

def train_model(stock_data_list, total_timesteps=200_000):
    print("Creating environment...")
    n_envs = 8
    # Create multiple environments running in parallel
    env = SubprocVecEnv(
        [
            make_env(
                stock_data_list,
                total_timesteps=total_timesteps,
                # rank=(i + 1) * 100_000,
                rank=(i + 1) * 10000,
                seed=MASTER_SEED,
            )
            for i in range(n_envs)
        ]
    )
    # env = VecNormalize(
    #     env,
    #     norm_obs=True,
    #     norm_reward=True,
    # )

    # env = PortfolioEnv(stock_data_list)
    # check_env(env)
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy", 
        env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/portfolio_env_logs",
        verbose=1,
        device="mps",
        n_steps=2048,
        learning_rate=1e-4,
        batch_size=128,
        # gamma=0.99,
        # ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128, 64], 
                          vf=[256, 256, 256, 128]),
            activation_fn=th.nn.Tanh
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=SAVE_DIR,
        name_prefix="ppov2",
        save_replay_buffer=False,
        save_vecnormalize=True,
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


def detailed_evaluation(trained_model, eval_env, n_episodes=10):
    # Result collection arrays
    all_allocations = []
    model_cumulative_returns = []
    model_monthly_returns = []
    model_sharpe_values = []
    model_max_drawdowns = []
    model_final_values = []
    
    benchmark_cumulative_returns = []
    benchmark_monthly_returns = []
    benchmark_sharpe_values = []
    benchmark_max_drawdowns = []
    benchmark_final_values = []
    
    # Arrays for visualization
    all_portfolio_curves = []
    all_benchmark_curves = []
    
    for episode in range(n_episodes):
        # Reset environment for new episode
        obs, info = eval_env.reset()
        
        # Initialize tracking variables
        model_value = 100.0
        benchmark_value = 100.0
        model_values = [model_value]
        benchmark_values = [benchmark_value]
        episode_allocations = []
        
        # Initialize return arrays for this episode
        model_returns = []
        benchmark_returns = []
        
        # Equal-weight allocation (10% each)
        n_stocks = eval_env.unwrapped.n_stocks
        equal_weight = np.ones(n_stocks) * (100 / n_stocks)
        
        # For forward-looking evaluation
        previous_allocation = equal_weight.copy()  # Start with equal weight
        done = False
        lstm_states = None
        step = 0
        
        while not done:
            # Get model's allocation decision
            action, lstm_states = trained_model.predict(obs, deterministic=True)
            
            # Save current allocation for performance evaluation in next period
            current_allocation = previous_allocation.copy()
            
            # Step environment to get next state and reward
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            
            # Get model's new allocation and stock returns (these are FUTURE returns)
            new_allocation = info['allocation'].copy()
            stock_returns = info['stock_returns']  # Price movements from this step to next
            
            # Store new allocation for next iteration
            previous_allocation = new_allocation
            episode_allocations.append(new_allocation)
            
            # Calculate this period's returns using CURRENT allocation (decided in previous step)
            # with CURRENT price movements (this step's stock_returns)
            model_return = np.sum((current_allocation / 100) * stock_returns)
            benchmark_return = np.sum((equal_weight / 100) * stock_returns)
            
            # Record returns for this step
            model_returns.append(model_return)
            benchmark_returns.append(benchmark_return)
            
            # Update portfolio values
            model_value *= (1 + model_return)
            benchmark_value *= (1 + benchmark_return)
            
            # Record portfolio values for visualization
            model_values.append(model_value)
            benchmark_values.append(benchmark_value)
        
        # Store complete episode data
        all_allocations.append(episode_allocations)
        all_portfolio_curves.append(model_values)
        all_benchmark_curves.append(benchmark_values)
        
        # Calculate episode metrics - EXACTLY THE SAME WAY for both portfolios
        
        # 1. Calculate average monthly return
        avg_model_return = np.mean(model_returns)
        avg_benchmark_return = np.mean(benchmark_returns)
        model_monthly_returns.append(avg_model_return)
        benchmark_monthly_returns.append(avg_benchmark_return)
        
        # 2. Calculate cumulative return
        model_cumulative_return = (model_value / 100.0) - 1
        benchmark_cumulative_return = (benchmark_value / 100.0) - 1
        model_cumulative_returns.append(model_cumulative_return)
        benchmark_cumulative_returns.append(benchmark_cumulative_return)
        
        # 3. Calculate Sharpe ratio (annualized)
        if len(model_returns) > 1:
            model_sharpe = np.mean(model_returns) / (np.std(model_returns) + 1e-10) * np.sqrt(12)
            benchmark_sharpe = np.mean(benchmark_returns) / (np.std(benchmark_returns) + 1e-10) * np.sqrt(12)
        else:
            model_sharpe = 0
            benchmark_sharpe = 0
        model_sharpe_values.append(model_sharpe)
        benchmark_sharpe_values.append(benchmark_sharpe)
        
        # 4. Calculate maximum drawdown
        model_max_dd = calculate_max_drawdown(model_values)
        benchmark_max_dd = calculate_max_drawdown(benchmark_values)
        model_max_drawdowns.append(model_max_dd)
        benchmark_max_drawdowns.append(benchmark_max_dd)
        
        # Store final values
        model_final_values.append(model_value)
        benchmark_final_values.append(benchmark_value)
        
        # Print episode summary
        print(f"Episode {episode+1}:")
        print(f"  Model:     Monthly Return: {avg_model_return*100:.2f}%, "
              f"Cumulative: {model_cumulative_return*100:.2f}%, "
              f"Final Value: ${model_value:.2f}")
        print(f"  Benchmark: Monthly Return: {avg_benchmark_return*100:.2f}%, "
              f"Cumulative: {benchmark_cumulative_return*100:.2f}%, "
              f"Final Value: ${benchmark_value:.2f}")
    
    # Calculate average metrics across all episodes
    avg_allocation = np.mean([np.mean(ep_allocs, axis=0) for ep_allocs in all_allocations], axis=0)
    
    # Average portfolio value curves for visualization
    avg_portfolio_curve = np.mean(all_portfolio_curves, axis=0)
    avg_benchmark_curve = np.mean(all_benchmark_curves, axis=0)
    
    # Create visualizations
    create_visualizations(
        avg_allocation,
        model_monthly_returns,
        model_sharpe_values,
        model_max_drawdowns,
        model_final_values,
        avg_portfolio_curve,
        avg_benchmark_curve,
        benchmark_monthly_returns,
        benchmark_sharpe_values,
        benchmark_max_drawdowns,
        benchmark_final_values
    )
    
    # Return consistent results
    return {
        'model_monthly_return': np.mean(model_monthly_returns),
        'model_cumulative_return': np.mean(model_cumulative_returns),
        'model_sharpe': np.mean(model_sharpe_values),
        'model_max_drawdown': np.mean(model_max_drawdowns),
        'model_final_value': np.mean(model_final_values),
        'avg_allocation': avg_allocation,
        'benchmark_monthly_return': np.mean(benchmark_monthly_returns),
        'benchmark_cumulative_return': np.mean(benchmark_cumulative_returns),
        'benchmark_sharpe': np.mean(benchmark_sharpe_values),
        'benchmark_max_drawdown': np.mean(benchmark_max_drawdowns),
        'benchmark_final_value': np.mean(benchmark_final_values)
    }


def calculate_max_drawdown(values):
    """Helper function to calculate maximum drawdown from a series of values"""
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / peaks
    return np.max(drawdowns)


def create_visualizations(avg_allocation, model_returns, model_sharpes, model_drawdowns,
                         model_final_values, portfolio_curve, benchmark_curve,
                         benchmark_returns, benchmark_sharpes, benchmark_drawdowns,
                         benchmark_final_values):
    RESULTS_DIR = f'{SAVE_DIR}/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Portfolio Allocation Visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(avg_allocation)), avg_allocation)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Stock')
    plt.ylabel('Average Allocation (%)')
    plt.title('Average Portfolio Allocation')
    plt.xticks(range(len(avg_allocation)), [f'Stock {i}' for i in range(len(avg_allocation))])
    plt.ylim(0, max(avg_allocation) * 1.2)
    plt.savefig(f'{RESULTS_DIR}/portfolio_allocation.png')
    plt.close()
    
    # 2. Portfolio Growth Curves Comparison
    plt.figure(figsize=(12, 6))
    months = range(len(portfolio_curve))
    plt.plot(months, portfolio_curve, 'b-', linewidth=2, label='Model Portfolio')
    plt.plot(months, benchmark_curve, 'r--', linewidth=2, label='Equal-Weight Benchmark')
    plt.xlabel('Month')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Forward-Looking Portfolio Growth Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{RESULTS_DIR}/forward_portfolio_growth.png')
    plt.close()
    
    # 3. Comparative Performance Metrics
    plt.figure(figsize=(12, 6))
    
    # Same calculation method for both values
    model_monthly_return = np.mean(model_returns) * 100
    benchmark_monthly_return = np.mean(benchmark_returns) * 100
    
    model_sharpe = np.mean(model_sharpes)
    benchmark_sharpe = np.mean(benchmark_sharpes)
    
    model_max_dd = np.mean(model_drawdowns) * 100
    benchmark_max_dd = np.mean(benchmark_drawdowns) * 100
    
    metrics = ['Monthly Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    model_values = [model_monthly_return, model_sharpe, model_max_dd]
    benchmark_values = [benchmark_monthly_return, benchmark_sharpe, benchmark_max_dd]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, model_values, width, label='Model Portfolio', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, benchmark_values, width, label='Equal-Weight Benchmark', color='red', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Forward-Looking Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels to bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'+{height:.2f}' if height >= 0 else f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'+{height:.2f}' if height >= 0 else f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.savefig(f'{RESULTS_DIR}/comparative_metrics.png')
    plt.close()
    
    # 4. Outperformance Summary
    plt.figure(figsize=(10, 6))
    outperformance = [
        model_monthly_return - benchmark_monthly_return,
        model_sharpe - benchmark_sharpe,
        benchmark_max_dd - model_max_dd  # For drawdown, lower is better
    ]
    
    colors = ['green' if val >= 0 else 'red' for val in outperformance]
    plt.bar(metrics, outperformance, color=colors, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Model Outperformance vs Equal-Weight (% difference)')
    
    for i, v in enumerate(outperformance):
        sign = '+' if v >= 0 else ''
        plt.text(i, v + 0.1 if v >= 0 else v - 0.5, 
                 f'{sign}{v:.2f}%',
                 ha='center', va='bottom' if v >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/outperformance_summary.png')
    plt.close()
    
    # 5. Final Values Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(model_final_values, bins=10, alpha=0.7, label='Model Portfolio')
    plt.hist(benchmark_final_values, bins=10, alpha=0.5, label='Equal-Weight Benchmark')
    
    plt.axvline(np.mean(model_final_values), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(benchmark_final_values), color='red', linestyle='dashed', linewidth=2)
    
    plt.text(np.mean(model_final_values)*1.02, plt.ylim()[1]*0.9, 
             f'Model Mean: ${np.mean(model_final_values):.2f}', color='blue')
    plt.text(np.mean(benchmark_final_values)*1.02, plt.ylim()[1]*0.8, 
             f'Benchmark Mean: ${np.mean(benchmark_final_values):.2f}', color='red')
    
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Portfolio Values')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/final_values_distribution.png')
    plt.close()
    
    print("Visualizations saved to 'results' directory")


def evaluate_model(stock_data_list, trained_model, n_episodes=10):
    print(f"Evaluating agent over {n_episodes} episodes with forward-looking metrics...")
    eval_env = Monitor(PortfolioEnv(stock_data_list, 
                                    mode="test",
                                    episode_length=58,
                                    seed=MASTER_SEED
                                    ))
    
    # Run the standard evaluation for baseline metrics
    mean_reward, std_reward = evaluate_policy(
        trained_model, 
        eval_env, 
        deterministic=True,
        n_eval_episodes=3 
    )
    print(f"Standard policy evaluation - Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    # Run our detailed forward-looking evaluation
    print(f"Running forward-looking evaluation...")
    results = detailed_evaluation(trained_model, eval_env, n_episodes)
    
    # Display performance summary
    print("\nForward-Looking Performance Summary:")
    print(f"Average Monthly Return: {results['model_monthly_return']*100:.2f}%")
    print(f"Cumulative Return: {results['model_cumulative_return']*100:.2f}%")
    print(f"Average Sharpe Ratio: {results['model_sharpe']:.2f}")
    print(f"Average Max Drawdown: {results['model_max_drawdown']*100:.2f}%")
    print(f"Final Average Portfolio Value: ${results['model_final_value']:.2f}")
    
    print("\nAverage Portfolio Allocation:")
    for i, alloc in enumerate(results['avg_allocation']):
        print(f"  Stock {i}: {alloc:.1f}%")
    
    # Display benchmark comparison
    print("\nForward-Looking Benchmark Comparison:")
    print(f"┌─────────────────┬────────────┬─────────────────┬────────────┐")
    print(f"│ Metric          │ Model      │ Equal-Weight    │ Difference │")
    print(f"├─────────────────┼────────────┼─────────────────┼────────────┤")
    
    # Monthly Return
    model_ret = results['model_monthly_return']*100
    bench_ret = results['benchmark_monthly_return']*100
    diff_ret = model_ret - bench_ret
    sign_ret = "+" if diff_ret > 0 else ""
    print(f"│ Monthly Return  │ {model_ret:8.2f}% │ {bench_ret:10.2f}% │ {sign_ret}{diff_ret:8.2f}% │")
    
    # Cumulative Return
    model_cum = results['model_cumulative_return']*100
    bench_cum = results['benchmark_cumulative_return']*100
    diff_cum = model_cum - bench_cum
    sign_cum = "+" if diff_cum > 0 else ""
    print(f"│ Cumulative Ret. │ {model_cum:8.2f}% │ {bench_cum:10.2f}% │ {sign_cum}{diff_cum:8.2f}% │")
    
    # Sharpe
    model_sharpe = results['model_sharpe']
    bench_sharpe = results['benchmark_sharpe']
    diff_sharpe = model_sharpe - bench_sharpe
    sign_sharpe = "+" if diff_sharpe > 0 else ""
    print(f"│ Sharpe Ratio    │ {model_sharpe:8.2f}  │ {bench_sharpe:10.2f}  │ {sign_sharpe}{diff_sharpe:8.2f}  │")
    
    # Drawdown
    model_dd = results['model_max_drawdown']*100
    bench_dd = results['benchmark_max_drawdown']*100
    diff_dd = bench_dd - model_dd
    sign_dd = "+" if diff_dd > 0 else ""
    print(f"│ Max Drawdown    │ {model_dd:8.2f}% │ {bench_dd:10.2f}% │ {sign_dd}{diff_dd:8.2f}% │")
    
    # Final Value
    model_val = results['model_final_value']
    bench_val = results['benchmark_final_value']
    diff_val = model_val - bench_val
    sign_val = "+" if diff_val > 0 else ""
    print(f"│ Final Value     │ ${model_val:7.2f}  │ ${bench_val:9.2f}  │ {sign_val}${diff_val:7.2f}  │")
    print(f"└─────────────────┴────────────┴─────────────────┴────────────┘")
    
    # Overall assessment
    if abs(diff_val) < 0.01:
        print(f"\nConclusion: Model performance is identical to the equal-weight benchmark")
        print(f"(This is expected since the model is currently using equal-weight allocations)")
    elif model_val > bench_val:
        outperf = ((model_val / bench_val) - 1) * 100
        print(f"\nConclusion: Model outperformed equal-weight benchmark by {outperf:.2f}%")
    else:
        underperf = ((bench_val / model_val) - 1) * 100
        print(f"\nConclusion: Model underperformed equal-weight benchmark by {underperf:.2f}%")
    
    return results


if __name__ == "__main__":
    # TRAIN
    instrument_list = ["AAPL", "MSFT", "JNJ", "PG", "JPM", "NVDA", "AMD", "TSLA", "CRM", "AMZN"] 
    stock_data_list = get_stock_data_list(instrument_list)
    print("Training model...")
    trained_model = train_model(stock_data_list, total_timesteps=1_000_000)
    print("Training complete!")

    # EVALUATE
    eval_instrument_list = ["IWM", "GS", "NEE", "PFE", "QQQ", "SPY", "VZ", "WMT", "XLE", "XLF"]        
    eval_stock_data_list = get_stock_data_list(eval_instrument_list)
    print("Evaluating model...")
    evaluate_model(eval_stock_data_list, trained_model)
    print("Evaluation complete!")
    # Conclusion: Model outperformed equal-weight benchmark by 13.15%