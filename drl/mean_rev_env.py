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
    def __init__(self, datasets, lookback_window=180, max_episode_steps=100):
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
            'bollinger_upper',
            'bollinger_lower',
            'bollinger_mid',
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

        # Process each feature
        for i, feature in enumerate(self.features):
            feature_values = window_data[feature].astype(float).values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_values = scaler.fit_transform(feature_values)

            # Add to observations
            observations[:, i] = scaled_values.flatten()
        # Add position state as the last feature for all timesteps
        position_value = 0.0  # No position
        if self.position == 'LONG':
            position_value = 1.0
        elif self.position == 'SHORT':
            position_value = -1.0

        observations[:, -1] = position_value
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
        match_reward = 1.0 if action == expert_action else -1.0
        flow_reward = 0.5 if self._check_valid_transition(action) else -1.0
        total_reward = match_reward + flow_reward
        return total_reward

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
        env = TradingImitationEnv(datasets)
        env = Monitor(env)
        return env
    return _init

def train_imitation_model(datasets, total_timesteps=1000000, n_envs=4):
    # Create vectorized environment
    env_fns = [make_env(datasets, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=SAVE_DIR,
        name_prefix="imitation_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Initialize model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/mean_rev_env_logs",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
            activation_fn=th.nn.Tanh,
            lstm_hidden_size=128,  # Size of LSTM hidden states
            n_lstm_layers=1, 
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
        action, lstm_states = model.predict(
            obs, 
            deterministic=True,
            state=lstm_states, 
            episode_start=np.array([done])
        )
        
        
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
    model = train_imitation_model(datasets, total_timesteps=500_000)
    
    # Evaluate on one of the datasets
    eval_symbol = 'CRM'
    actions, expert_actions, accuracy = evaluate_imitation(model, datasets, eval_symbol)
