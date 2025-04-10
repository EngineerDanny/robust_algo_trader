import numpy as np
import pandas as pd
import datetime as dt
import os
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import torch as th
from sb3_contrib import RecurrentPPO

SAVE_DIR = f"/Users/newuser/Projects/robust_algo_trader/drl/models/model_{dt.datetime.now().strftime('%Y%m%d_%H')}"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = "/Users/newuser/Projects/robust_algo_trader/data/gen_alpaca_data"
MASTER_SEED = 42


class TradingProfitEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, signals_data, lookback_window=180, max_steps_per_trade=100,
                 commission_rate=0.002):
        super(TradingProfitEnv, self).__init__()

        self.signals_data = signals_data
        self.symbols = list(signals_data.keys())
        self.current_symbol = None
        self.lookback_window = lookback_window
        self.max_steps_per_trade = max_steps_per_trade
        self.commission_rate = commission_rate
        
        self.features = [
            'distance_from_mean',
            # 'distance_from_upper',
            # 'distance_from_lower',
            'rsi',
            'range_strength',
            'mean_reversion_probability',
            # 'is_range_market'
        ]
        
        self.action_map = {
            'HOLD': 0,
            'CLOSE': 1
        }
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        self.action_space = spaces.Discrete(len(self.action_map))
        
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(lookback_window, len(self.features) + 1),
            dtype=np.float32
        )

        self.position = None
        self.position_type = None
        self.position_entry_step = None
        self.entry_price = None
        self.steps_in_trade = 0
        self.current_step = None
        self.trade_history = []
        
    def _find_next_entry_signal(self):
        signals = self.signals_data[self.current_symbol]
       
        if self.current_step is None:
            start_idx = self.lookback_window
        else:
            # Otherwise, start from the step after current position
            start_idx = self.current_step + 1
        
        # Look for the next entry signal in sequential order
        for i in range(start_idx, len(signals) - 1):
            action = signals.iloc[i]['action']
            if action in ['BUY', 'SELL']:
                self.current_step = i
                self.position_type = action
                self.position = 'LONG' if action == 'BUY' else 'SHORT'
                self.position_entry_step = i
                self.entry_price = signals.iloc[i]['price']
                return
        
        # If no more signals are found, set a flag to indicate end of data
        self.end_of_data = True
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Only change symbol when explicitly told to or at the beginning
        if self.current_symbol is None:
            self.current_symbol = random.choice(self.symbols)
        
        self.position = None
        self.position_type = None
        self.position_entry_step = None
        self.entry_price = None
        self.steps_in_trade = 0
        self.end_of_data = False
        
        # Find next entry signal (sequential)
        self._find_next_entry_signal()
        
        # If we've reached the end of data, wrap around to the beginning
        if hasattr(self, 'end_of_data') and self.end_of_data:
            self.current_step = self.lookback_window
            self.end_of_data = False
            self._find_next_entry_signal()
        
        return self._get_observation(), {}

    def _get_observation(self):
        signals = self.signals_data[self.current_symbol]

        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        window_data = signals.iloc[start_idx:end_idx]

        observations = np.zeros((self.lookback_window, len(self.features) + 1), dtype=np.float32)

        for i, feature in enumerate(self.features):
            window_values = window_data[feature].astype(float).values
            
            if feature == 'rsi':
                scaled_values = 2 * (window_values / 100) - 1
            elif feature == 'mean_reversion_probability':
                scaled_values = 2 * window_values - 1
            elif feature == 'is_range_market':
                scaled_values = 2 * window_values - 1
            elif feature in ['distance_from_mean']:
                scaled_values = window_values / 10
            elif feature in ['distance_from_upper', 'distance_from_lower']:
                scaled_values = 2 * (window_values / 100) - 1
            elif feature == 'range_strength':
                scaled_values = 2 * window_values - 1
            
            scaled_values = np.clip(scaled_values, -1, 1)
            observations[:, i] = scaled_values

        observations[:, -1] = self._get_position_state()
        return observations.astype(np.float32)

    def _get_position_state(self):
        if self.position is None:
            return 0.0
        elif self.position == 'LONG':
            return 1.0
        else:
            return -1.0
    
    def _calculate_unrealized_pnl(self, current_price):
        if self.position is None or self.entry_price is None:
            return 0.0
        pnl = 0.0
        if self.position == 'LONG':
            pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == 'SHORT':
            pnl = (self.entry_price - current_price) / self.entry_price
            
        return pnl
    
    def _calculate_reward(self, action, current_price):
        action_str = self.reverse_action_map[action]
        
        if action_str == 'CLOSE':
            if self.position == 'LONG':
                pnl = np.log(current_price / self.entry_price)
                pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == 'SHORT':
                pnl = np.log(self.entry_price / current_price)
                pnl = (self.entry_price - current_price) / self.entry_price
            else:
                return 0.0
                
            total_costs = np.log(self.entry_price * self.commission_rate)
            
            total_costs = self.commission_rate
            net_pnl = pnl - total_costs
            reward = net_pnl * 100
            # Bonus for profitable trades
            # if net_pnl > 0:
            #     reward += 1.0
                
            self.trade_history.append({
                'symbol': self.current_symbol,
                'position': self.position,
                'entry_step': self.position_entry_step,
                'exit_step': self.current_step,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'net_pnl': net_pnl,
                'costs': total_costs,
                'duration': self.current_step - self.position_entry_step
            })
            
            return reward
        else:  # HOLD action
            return 0
            hold_reward = 0
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            if unrealized_pnl > 0:
                hold_reward += unrealized_pnl * 5  # Significant scaling
            return hold_reward

    def step(self, action):
        signals = self.signals_data[self.current_symbol]
        current_price = signals.iloc[self.current_step]['price']
        done = (self.steps_in_trade >= self.max_steps_per_trade) or (self.current_step >= len(signals) - 1)
        if done:
            action = self.action_map['CLOSE']
        
        
        reward = self._calculate_reward(action, current_price)
        action_str = self.reverse_action_map[action]
        
        if action_str == 'CLOSE':
            self.position = None
            self.position_type = None
            self.position_entry_step = None
            self.entry_price = None
            done = True
        else:
            self.current_step += 1
            self.steps_in_trade += 1
        obs = self._get_observation()
        
        info = {
            "Reward": reward,
            "Position": self.position,
            "Position Type": self.position_type,
            "Action": action_str,
            "Current Price": current_price,
            "Entry Price": self.entry_price,
            "Current Symbol": self.current_symbol,
            "Steps in Trade": self.steps_in_trade,
            "Unrealized PnL": self._calculate_unrealized_pnl(current_price)
        }
        
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def make_env(signals_data, rank):
    def _init():
        env = Monitor(TradingProfitEnv(signals_data))
        return env
    return _init


def train_profit_model(signals_data, total_timesteps=1_000_000, n_envs=8):
    env_fns = [make_env(signals_data, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=SAVE_DIR,
        name_prefix="profit_model",
        # save_vecnormalize=True,
    )
    
    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/mean_rev_env_logs",
    #     learning_rate=3e-4,
    #     n_steps=1024,
    #     batch_size=64,
    #     n_epochs=5,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     verbose=1,
    #     policy_kwargs=dict(
    #         net_arch=[256, 128, 64],
    #         activation_fn=th.nn.Tanh,
    #     )
    # )
    
    print("Initializing RecurrentPPO agent...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        vec_env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/mean_rev_env_logs",
        verbose=1,
        # device="mps",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        # ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
            activation_fn=th.nn.Tanh,
            lstm_hidden_size=128,  # Size of LSTM hidden states
            n_lstm_layers=1, 
        )
    )

    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    final_model_path = os.path.join(SAVE_DIR, "final_profit_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    return model



if __name__ == "__main__":
    signals_data = {
        'CRM': pd.read_csv(os.path.join(DATA_DIR, 'CRM_M1_signals.csv'))
    }
    
    model = train_profit_model(signals_data, total_timesteps=2_000_000)
    
    symbol = 'CRM'
    eval_data = {symbol: signals_data[symbol]}
    eval_env = Monitor(TradingProfitEnv(eval_data))
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")