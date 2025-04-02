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

# save dir should add the current date and time
SAVE_DIR = f"/Users/newuser/Projects/robust_algo_trader/drl/models/model_{dt.datetime.now().strftime('%Y%m%d_%H')}"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_alpaca_data"
MASTER_SEED = 42


class TradingImitationEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, datasets, lookback_window=180, max_episode_steps=1000):
        super(TradingImitationEnv, self).__init__()

        # Store datasets (dictionary of {symbol: {'data': df, 'actions': series}})
        self.datasets = datasets
        self.symbols = list(datasets.keys())
        self.current_symbol = random.choice(self.symbols)
        self.lookback_window = lookback_window

        # Define features we want to use from the dataset
        self.features = [
            'distance_from_mean',
            'distance_from_upper',
            'distance_from_lower',
            'rsi',
            # 'bollinger_upper',
            # 'bollinger_lower',
            # 'bollinger_mid',
            'range_strength',
            'mean_reversion_probability',
            'is_range_market'
        ]
        # Action mapping
        self.action_map = {
            'NOTHING': 0,
            'HOLD': 1,
            'CLOSE': 2,
            'BUY': 3,
            'SELL': 4
        }
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        self.action_space = spaces.Discrete(len(self.action_map))
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(lookback_window, len(self.features) + 1),
            dtype=np.float32
        )

        self.current_step = self.lookback_window + 60
        self.position = None
        self.position_entry_step = None
        self.last_action = None
        self.max_episode_steps = max_episode_steps
        self.steps_in_episode = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose a random dataset for this episode
        self.current_symbol = random.choice(self.symbols)
        self.position = None
        self.position_entry_step = None
        self.last_action = None
        self.steps_in_episode = 0

        return self._get_observation(), {}

    def _get_observation(self):
        # Get current dataset
        current_data = self.datasets[self.current_symbol]['data']

        # Extract lookback window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        window_data = current_data.iloc[start_idx:end_idx]

        # Create features array [lookback_window, features]
        observations = np.zeros((self.lookback_window, len(self.features) + 1), dtype=np.float32)

        # Process each feature with appropriate scaling
        for i, feature in enumerate(self.features):
            window_values = window_data[feature].astype(float).values
            
            # Apply appropriate scaling based on feature type
            if feature == 'rsi':
                # RSI is already 0-100, scale to -1 to 1
                scaled_values = 2 * (window_values / 100) - 1
            elif feature == 'mean_reversion_probability':
                # Assuming this is 0-1 probability, scale to -1 to 1
                scaled_values = 2 * window_values - 1
            elif feature == 'is_range_market':
                # Boolean feature, convert to -1 or 1
                # Assuming True/False or 1/0
                scaled_values = 2 * window_values - 1
            elif feature in ['distance_from_mean']:
                # These are already relative distances, may need capping
                scaled_values = window_values / 10
            # For percentage features (0-100 range like distance_from_upper)
            elif feature in ['distance_from_upper', 'distance_from_lower']:
                # Simple linear scaling from [0,100] to [-1,1]
                scaled_values = 2 * (window_values / 100) - 1
            # elif feature in ['bollinger_upper', 'bollinger_lower', 'bollinger_mid']:
            #     # Scale price-based indicators relative to the mid price
            #     if 'bollinger_mid' in window_data.columns:
            #         mid_prices = window_data['bollinger_mid'].astype(float).values
            #         scaled_values = (window_values - mid_prices) / mid_prices
            #         # Cap at ±10% which is typical for Bollinger Bands
            #         scaled_values = np.clip(scaled_values * 10, -1, 1)
            #     else:
            #         # Fallback if mid price not available
            #         scaled_values = np.zeros_like(window_values)
            elif feature == 'range_strength':
                scaled_values = 2 * window_values - 1
               
            
            # Ensure all values are within [-1, 1]
            scaled_values = np.clip(scaled_values, -1, 1)
            observations[:, i] = scaled_values

        # Position state remains unchanged (already -1 to 1)
        observations[:, -1] = self._get_position_state()
        return observations.astype(np.float32)

    def _get_position_state(self):
        if self.position is None:
            return 0.0
        elif self.position == 'LONG':
            return 1.0
        else:  # SHORT
            return -1.0

    def get_expert_action(self):
        expert_actions = self.datasets[self.current_symbol]['actions']
        # Handle both DataFrame and Series cases
        action_str = expert_actions.iloc[self.current_step]['action']
        return self.action_map.get(action_str, 0) 
    

    def _check_valid_transition(self, action):
        action_str = self.reverse_action_map[action]

        # Valid transitions
        if self.position is None:  # No position
            if action_str in ['NOTHING', 'BUY', 'SELL']:
                return True
            return False

        elif self.position == 'LONG':  # Long position
            if action_str in ['HOLD', 'CLOSE']:
                return True
            return False

        elif self.position == 'SHORT':  # Short position
            if action_str in ['HOLD', 'CLOSE']:
                return True
            return False

        return False
    
    def _calculate_reward(self, action):
        expert_action = self.get_expert_action()
        expert_action_str = self.reverse_action_map[expert_action]
        
        # Inverse frequency weights (adjusted for relative importance)
        action_weights = {
            'NOTHING': 0.2,   # 1/(0.79*5) ≈ 0.2, downweighted common action
            'HOLD': 0.6,      # 1/(0.21*5) ≈ 0.6, slightly downweighted
            'CLOSE': 5.0,     # 1/0.04 = 25, but adjusted to 5
            'BUY': 6.7,       # 1/0.03 ≈ 33, but adjusted to 6.7
            'SELL': 10.0      # 1/0.01 = 100, but adjusted to 10
        }
        
        # Calculate reward based on whether prediction matches expert
        if action == expert_action:
            match_reward = action_weights.get(expert_action_str, 1.0)
        else:
            # For mismatched predictions, penalize less for common actions
            # and more for missing rare actions
            if expert_action_str in ['BUY', 'SELL', 'CLOSE']:
                miss_penalty = -2.0  # Stronger penalty for missing important signals
            else:
                miss_penalty = -0.5  # Lighter penalty for missing NOTHING/HOLD
            match_reward = miss_penalty
        return match_reward

    # def _calculate_reward(self, action):
    #     expert_action = self.get_expert_action()
    #     match_reward = 1.0 if action == expert_action else -1.0
    #     # flow_reward = 0.5 if self._check_valid_transition(action) else -1.0
    #     flow_reward = 0
    #     total_reward = match_reward + flow_reward
    #     return total_reward

    def step(self, action):
        reward = self._calculate_reward(action)
        action_str = self.reverse_action_map[action]

        if action_str == 'BUY':
            self.position = 'LONG'
            self.position_entry_step = self.current_step
        elif action_str == 'SELL':
            self.position = 'SHORT'
            self.position_entry_step = self.current_step
        elif action_str == 'CLOSE':
            self.position = None
            self.position_entry_step = None

        # Store last action
        self.last_action = action_str
        self.current_step += 1
        self.steps_in_episode += 1

        current_data = self.datasets[self.current_symbol]['data']
        truncated = self.current_step >= len(current_data) - 1
        terminated = self.steps_in_episode >= self.max_episode_steps
        done = terminated or truncated
        
        if truncated:
            self.current_step = self.lookback_window + 60

        obs = self._get_observation()
        info = {
            "Reward": reward,
            "Position": self.position,
            "Action": action_str,
            "Current Symbol": self.current_symbol,
            "Done": done
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def make_env(datasets, rank):
    def _init():
        env = Monitor(TradingImitationEnv(datasets))
        return env
    return _init

def train_imitation_model(datasets, total_timesteps=1_000_000, n_envs=8):
    # Create vectorized environment
    env_fns = [make_env(datasets, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    
    # vec_env = Monitor(TradingImitationEnv(datasets))
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=SAVE_DIR,
        name_prefix="imitation_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Initialize model
    # model = RecurrentPPO(
    model = PPO(
        # "MlpLstmPolicy",
        "MlpPolicy",
        vec_env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/mean_rev_env_logs",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
            activation_fn=th.nn.Tanh,
            # lstm_hidden_size=256,
            # n_lstm_layers=1, 
        )
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, "final_imitation_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    return model

def evaluate_imitation(model, datasets, symbol, n_eval_episodes=10):
    # Create single environment for evaluation
    eval_datasets = {symbol: datasets[symbol]}
    eval_env = Monitor(TradingImitationEnv(eval_datasets))
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Collect model actions for comparison
    obs, _ = eval_env.reset()
    done = False
    actions = []
    expert_actions = []
    lstm_states = None
    
    while not done:
        # action, lstm_states = model.predict(
        #     obs, 
        #     deterministic=True,
        #     state=lstm_states, 
        #     episode_start=np.array([done])
        # )

        # In your evaluation function
        action, _ = model.predict(obs, deterministic=True)
        
        
        action = int(action)
        expert_action = eval_env.unwrapped.get_expert_action()
        
        actions.append(eval_env.unwrapped.reverse_action_map[action])
        expert_actions.append(eval_env.unwrapped.reverse_action_map[expert_action])
        
        obs, _, done, _, _ = eval_env.step(action)
        if done:
            break
    
    # Compare actions
    correct = sum(1 for a, e in zip(actions, expert_actions) if a == e)
    total = len(actions)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Action matching accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Print confusion matrix
    action_types = ['NOTHING', 'HOLD', 'CLOSE', 'BUY', 'SELL']
    confusion = np.zeros((len(action_types), len(action_types)), dtype=int)
    
    for pred, true in zip(actions, expert_actions):
        pred_idx = action_types.index(pred)
        true_idx = action_types.index(true)
        confusion[true_idx, pred_idx] += 1
    
    print("\nConfusion Matrix (rows: true, cols: predicted):")
    print("               " + " ".join(f"{a:>8}" for a in action_types))
    for i, a in enumerate(action_types):
        print(f"{a:>8}", end="")
        for j in range(len(action_types)):
            print(f"{confusion[i, j]:>9}", end="")
        print()
    
    return actions, expert_actions, accuracy

# Example usage
if __name__ == "__main__":
    
    datasets = {
        'CRM': {
            'data': pd.read_csv(os.path.join(DATA_DIR, 'CRM_M1_train_data.csv')), 
            'actions': pd.read_csv(os.path.join(DATA_DIR, 'CRM_M1_signals.csv'))   
        }
    }
    
    # Train model
    model = train_imitation_model(datasets, total_timesteps=1_000_000)
    
    # Evaluate on one of the datasets
    eval_symbol = 'CRM'
    actions, expert_actions, accuracy = evaluate_imitation(model, datasets, eval_symbol)
