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

SAVE_DIR = f"./models/model_{dt.datetime.now().strftime('%Y%m%d_%H')}"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = "./data/gen_alpaca_data"
MASTER_SEED = 42


class TradingProfitEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, signals_data, lookback_window=180, max_steps_per_trade=500,
                 commission_rate=0.001, slippage_factor=0.0005):
        super(TradingProfitEnv, self).__init__()

        self.signals_data = signals_data
        self.symbols = list(signals_data.keys())
        self.current_symbol = None
        self.lookback_window = lookback_window
        self.max_steps_per_trade = max_steps_per_trade
        
        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor
        
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
        self.max_adverse_move = 0
        self.steps_in_trade = 0
        self.current_step = None
        self.trade_history = []
        
    def _find_next_entry_signal(self):
        signals = self.signals_data[self.current_symbol]
        start_idx = self.lookback_window + random.randint(0, 100)
        
        for i in range(start_idx, len(signals) - self.max_steps_per_trade):
            action = signals.iloc[i]['action']
            if action in ['BUY', 'SELL']:
                self.current_step = i
                self.position_type = action
                self.position = 'LONG' if action == 'BUY' else 'SHORT'
                self.position_entry_step = i
                self.entry_price = signals.iloc[i]['price']
                break
        
        if self.current_step is None:
            for i in range(self.lookback_window, len(signals) - self.max_steps_per_trade):
                action = signals.iloc[i]['action']
                if action in ['BUY', 'SELL']:
                    self.current_step = i
                    self.position_type = action
                    self.position = 'LONG' if action == 'BUY' else 'SHORT'
                    self.position_entry_step = i
                    self.entry_price = signals.iloc[i]['price']
                    break
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_symbol = random.choice(self.symbols)
        self.position = None
        self.position_type = None
        self.position_entry_step = None
        self.entry_price = None
        self.max_adverse_move = 0
        self.steps_in_trade = 0
        
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
                pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == 'SHORT':
                pnl = (self.entry_price - current_price) / self.entry_price
            else:
                return 0.0
                
            entry_commission = self.entry_price * self.commission_rate
            entry_slippage = self.entry_price * self.slippage_factor
            exit_commission = current_price * self.commission_rate
            total_costs = (entry_commission + entry_slippage + exit_commission) / self.entry_price
            
            net_pnl = pnl - total_costs
            
            # Simple reward based on scaled net P&L
            reward = net_pnl * 25
            
            # Bonus for profitable trades
            if net_pnl > 0:
                reward += 1.0
                
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
                'max_drawdown': self.max_adverse_move,
                'duration': self.current_step - self.position_entry_step
            })
            
            return reward

        else:  # HOLD action
            # Neutral starting point
            hold_reward = 0.0
            
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            if unrealized_pnl < 0 and abs(unrealized_pnl) > self.max_adverse_move:
                self.max_adverse_move = abs(unrealized_pnl)
                
            # Stronger incentive for holding profitable positions
            if unrealized_pnl > 0.01:
                hold_reward += 0.05  # Much larger reward (25x increase)
            
            # Add a small positive bias to encourage exploration
            hold_reward += 0.001
                
            return hold_reward

    def step(self, action):
        signals = self.signals_data[self.current_symbol]
        current_price = signals.iloc[self.current_step]['price']
        
        reward = self._calculate_reward(action, current_price)
        
        action_str = self.reverse_action_map[action]
        
        # This is the important part:
        # Set done=True when action is CLOSE
        done = False
        
        if action_str == 'CLOSE':
            self.position = None
            self.position_type = None
            self.position_entry_step = None
            self.entry_price = None
            self.max_adverse_move = 0
            
            # Set done=True to end the episode
            done = True
        else:
            self.current_step += 1
            self.steps_in_trade += 1
            # Also end episode if max steps reached or end of data
            done = (self.steps_in_trade >= self.max_steps_per_trade) or (self.current_step >= len(signals) - 1)
        
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
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=SAVE_DIR,
        name_prefix="profit_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        tensorboard_log="/Users/newuser/Projects/robust_algo_trader/drl/mean_rev_env_logs",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
            activation_fn=th.nn.Tanh,
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


def evaluate_profit_model(model, signals_data, symbol, n_eval_episodes=20):
    eval_data = {symbol: signals_data[symbol]}
    eval_env = Monitor(TradingProfitEnv(eval_data))
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    all_trades = eval_env.unwrapped.trade_history
    
    if all_trades:
        total_trades = len(all_trades)
        profitable_trades = sum(1 for t in all_trades if t['net_pnl'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([t['net_pnl'] for t in all_trades if t['net_pnl'] > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([t['net_pnl'] for t in all_trades if t['net_pnl'] <= 0]) if total_trades - profitable_trades > 0 else 0
        
        profit_factor = abs(sum([t['net_pnl'] for t in all_trades if t['net_pnl'] > 0])) / abs(sum([t['net_pnl'] for t in all_trades if t['net_pnl'] < 0])) if sum([t['net_pnl'] for t in all_trades if t['net_pnl'] < 0]) != 0 else float('inf')
        
        avg_trade_duration = np.mean([t['duration'] for t in all_trades])
        
        print("\nTrading Performance Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Profit: {avg_profit:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Trade Duration: {avg_trade_duration:.1f} steps")
        
        cumulative_returns = np.cumsum([t['net_pnl'] for t in all_trades])
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns * 100)
        plt.title(f"Cumulative Returns for {symbol}")
        plt.xlabel("Trade Number")
        plt.ylabel("Cumulative Return (%)")
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, f"{symbol}_returns.png"))
        plt.show()
    
    return all_trades


if __name__ == "__main__":
    signals_data = {
        'CRM': pd.read_csv(os.path.join(DATA_DIR, 'CRM_M1_signals.csv'))
    }
    
    model = train_profit_model(signals_data, total_timesteps=1_000_000)
    
    eval_symbol = 'CRM'
    trade_history = evaluate_profit_model(model, signals_data, eval_symbol)