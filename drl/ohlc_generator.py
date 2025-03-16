import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import student_t  # Add this import for the student_t distribution
import talib  # Add this import for technical indicators

class SimpleOHLCGenerator:
    def __init__(self, original_data=None):
        self.original_data = None
        self.max_candle_size = None
        self.max_day_change = None
        
        if original_data is not None:
            self.original_data = self._preprocess_data(original_data)
            # Calculate volatility statistics from the original data
            # Calculate the maximum candle size from original data
            self.max_candle_size = (self.original_data['high'] - self.original_data['low']).max()
            
            # Calculate maximum day-to-day change
            self.original_data['prev_close'] = self.original_data['close'].shift(1)
            self.original_data['day_change'] = abs(self.original_data['open'] - self.original_data['prev_close'])
            self.max_day_change = self.original_data['day_change'].max()
    
    def _preprocess_data(self, data):
        df = data.copy()
        # Standardize column names and set datetime index
        df.columns = [col.lower() for col in df.columns]
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        # Round OHLC values to two decimals
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].round(2)
        df = df.sort_index()  # Ensure data is sorted by time
        return df

    def generate_bootstrap_data(self, num_samples=1, segment_length=5, days=2520):
        if self.original_data is None:
            raise ValueError("Original data must be provided for bootstrap data generation")
            
        data_length = len(self.original_data)
        segment_length = min(segment_length, data_length)
        segments = [self.original_data.iloc[i:i+segment_length].copy() 
                    for i in range(data_length - segment_length + 1)]
        
        num_segments_needed = days // segment_length + 1
        synthetic_datasets = []
        
        for _ in range(num_samples):
            sampled_segments = random.choices(segments, k=num_segments_needed)
            synthetic_data = self._stitch_segments(sampled_segments)
            
            # Remove any negative price candles
            synthetic_data = synthetic_data[(synthetic_data['open'] > 0) & 
                                            (synthetic_data['high'] > 0) & 
                                            (synthetic_data['low'] > 0) & 
                                            (synthetic_data['close'] > 0)]
            
            # Validation checks
            if len(synthetic_data) > 1:
                synthetic_datasets.append(synthetic_data)
 
        return synthetic_datasets

    def _stitch_segments(self, segments):
        if self.original_data is None:
            raise ValueError("Original data must be provided for segment stitching")
            
        # Start with the first segment
        result = segments[0].copy()
        freq = pd.infer_freq(self.original_data.index) or 'B'
        
        for seg in segments[1:]:
            seg = seg.copy()
            if len(result) == 0:  # Handle empty result case
                result = seg.copy()
                continue
                
            last_close = result['close'].iloc[-1]
            last_date = result.index[-1]
            
            # Generate new dates for this segment
            new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                      periods=len(seg),
                                      freq=freq)
            seg.index = new_dates
            
            # Strict Enforcement: First open exactly equals last close
            first_open = seg['open'].iloc[0]
            scaling_factor = last_close / first_open if first_open != 0 else 1.0
            
            # Scale the entire segment to maintain relative proportions
            for col in ['open', 'high', 'low', 'close']:
                seg[col] = seg[col] * scaling_factor
            
            # Strictly enforce first open equals last close
            seg.iloc[0, seg.columns.get_loc('open')] = last_close
            
            # Process each row in the segment
            for i in range(len(seg)):
                # For the first row, we already set the open
                if i > 0:
                    # Set open to previous close for continuity
                    prev_close = seg.iloc[i-1]['close']
                    seg.iloc[i, seg.columns.get_loc('open')] = prev_close
                
                # Get current values
                o = seg.iloc[i]['open']
                h = seg.iloc[i]['high']
                l = seg.iloc[i]['low']
                c = seg.iloc[i]['close']
                
                # Check if this candle is too large (high-low range)
                candle_size = h - l
                if candle_size > self.max_candle_size * 1.1:  # Allow 10% extra
                    # Replace with a reasonable candle size
                    # Center the new candle around the open
                    avg_candle_size = self.max_candle_size * 0.5  # Use half of max as typical
                    h = o + (avg_candle_size / 2)
                    l = o - (avg_candle_size / 2)
                    
                    # Set close to be in the same direction but with moderate movement
                    if c > o:
                        c = o + (avg_candle_size * 0.3)  # 30% of avg candle size up
                    else:
                        c = o - (avg_candle_size * 0.3)  # 30% of avg candle size down
                
                # Check if open-to-close move is too large
                if abs(c - o) > self.max_day_change * 1.1:  # Allow 10% extra
                    # Limit the close to a reasonable range from open
                    direction = 1 if c > o else -1
                    c = o + (direction * self.max_day_change * 0.8)  # 80% of max daily change
                
                # OHLC consistency check
                h = max(h, o, c)
                l = min(l, o, c)
                
                # Ensure positive values
                h = max(0.01, h)
                l = max(0.01, l)
                c = max(0.01, c)
                
                # Write back the adjusted values
                seg.iloc[i, seg.columns.get_loc('high')] = round(h, 2)
                seg.iloc[i, seg.columns.get_loc('low')] = round(l, 2)
                seg.iloc[i, seg.columns.get_loc('close')] = round(c, 2)
            
            # Final continuity check
            if seg.iloc[0]['open'] != result.iloc[-1]['close']:
                seg.iloc[0, seg.columns.get_loc('open')] = result.iloc[-1]['close']
            
            # Concatenate with the result
            result = pd.concat([result, seg])
        
        # Final pass to remove any extremely large candles that might have slipped through
        if len(result) > 0:
            # Create a temporary column for candle size
            result['candle_size'] = result['high'] - result['low']
            result['price_change'] = abs(result['close'] - result['open'])
            
            # Flag extreme candles (3x larger than max observed in original data)
            extreme_candles = (result['candle_size'] > self.max_candle_size * 3) | \
                              (result['price_change'] > self.max_day_change * 3)
            
            if extreme_candles.any():
                # Get indices of extreme candles
                extreme_indices = result.index[extreme_candles]
                
                # For each extreme candle
                for idx in extreme_indices:
                    # Get position in the DataFrame
                    pos = result.index.get_loc(idx)
                    
                    if pos > 0:  # Not the first row
                        # Get the row before
                        prev_row = result.iloc[pos-1]
                        prev_close = prev_row['close']
                        
                        # Create a simple replacement candle
                        result.loc[idx, 'open'] = prev_close
                        result.loc[idx, 'close'] = prev_close * np.random.uniform(0.99, 1.01)
                        result.loc[idx, 'high'] = max(result.loc[idx, 'open'], result.loc[idx, 'close']) * 1.005
                        result.loc[idx, 'low'] = min(result.loc[idx, 'open'], result.loc[idx, 'close']) * 0.995
            
            # Drop the temporary columns
            result = result.drop(columns=['candle_size', 'price_change'])
        
        return result

    def generate_synthetic_data(self, n_stocks=1, synthetic_data_years=10, 
                                min_sharpe=0.4, 
                                min_annual_return=0.06, 
                                max_attempts=20, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Market regime constants
        BULL = "bull"
        BEAR = "bear"
        CORRECTION = "correction"
        CRASH = "crash"
        RECOVERY = "recovery"
        
        # Transition probabilities - adjusted for more realistic long-term trends
        regime_transitions = {
            'bull_to_bear': 0.008,      
            'bull_to_correction': 0.03,  
            'bear_to_bull': 0.15,       
            'correction_length': (5, 12),
            'correction_depth': (-0.10, -0.03),
        }
        
        synthetic_data_list = []
        
        while len(synthetic_data_list) < n_stocks:
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                
                # Stock-specific parameters with randomization
                bull_drift = np.random.normal(0.14, 0.03) 
                bear_drift = np.random.normal(-0.10, 0.02)
                upward_bias = np.random.normal(0.09, 0.02) 
                bull_vol = max(np.random.normal(0.15, 0.03), 0.02)
                bear_vol = max(np.random.normal(0.25, 0.03), 0.05)
                
                # Generate dates
                trading_days = synthetic_data_years * 252
                dates = pd.date_range(
                    start=pd.Timestamp('2010-01-01'),
                    periods=trading_days,
                    freq='B'  # Business days
                )
                
                # Initialize arrays
                N = len(dates)
                close_prices = np.zeros(N)
                open_prices = np.zeros(N)
                high_prices = np.zeros(N)
                low_prices = np.zeros(N)
                regimes = np.array([BULL] * N, dtype=object)
                
                # Initial values
                initial_price = np.random.uniform(50, 150)
                close_prices[0] = initial_price
                open_prices[0] = initial_price * (1 + np.random.normal(0, 0.005))
                high_prices[0] = max(close_prices[0], open_prices[0]) * (1 + abs(np.random.normal(0, 0.01)))
                low_prices[0] = min(close_prices[0], open_prices[0]) * (1 - abs(np.random.normal(0, 0.01)))
                
                # Track regime state
                current_regime = BULL
                correction_target = None
                correction_end = None
                
                # Generate subsequent days
                for j in range(1, N):
                    # Update regime
                    r = np.random.random()
                    
                    if current_regime == BULL:
                        if r < regime_transitions['bull_to_bear']:
                            current_regime = BEAR
                        elif r < (regime_transitions['bull_to_bear'] + regime_transitions['bull_to_correction']):
                            current_regime = CORRECTION
                            dur = np.random.randint(*regime_transitions['correction_length'])
                            correction_end = j + dur
                            correction_target = np.random.uniform(*regime_transitions['correction_depth'])
                    elif current_regime == BEAR:
                        if r < regime_transitions['bear_to_bull']:
                            current_regime = RECOVERY
                            bear_days = np.sum(regimes[:j] == BEAR)
                            correction_end = j + min(int(bear_days * 0.5), 30)  # Cap recovery period
                    elif current_regime == CORRECTION:
                        if correction_end is not None and j >= correction_end:
                            current_regime = BULL
                            correction_target = None
                            correction_end = None
                    elif current_regime == RECOVERY:
                        if correction_end is not None and j >= correction_end:
                            current_regime = BULL
                            correction_end = None
                    elif current_regime == CRASH:
                        current_regime = RECOVERY
                        correction_end = j + 10  # Short recovery after crash
                        
                    regimes[j] = current_regime
                    
                    # Set drift and volatility based on regime
                    if current_regime == BULL:
                        drift = bull_drift
                        vol = bull_vol
                    elif current_regime == BEAR:
                        drift = bear_drift
                        vol = bear_vol
                    elif current_regime == CORRECTION:
                        drift = correction_target if correction_target is not None else np.random.uniform(*regime_transitions['correction_depth'])
                        vol = 0.5 * (bull_vol + bear_vol)
                    elif current_regime == RECOVERY:
                        drift = bull_drift * 1.5
                        vol = bull_vol + 0.3 * (bear_vol - bull_vol)
                    elif current_regime == CRASH:
                        drift = np.random.uniform(-0.15, -0.05)
                        vol = bear_vol * 2
                    else:
                        drift = bull_drift
                        vol = bull_vol
                        
                    # Convert annual to daily
                    daily_drift = np.log(1 + drift) / 252
                    daily_drift += upward_bias / 252
                    daily_vol = vol / np.sqrt(252)
                    
                    # Generate returns with t-distribution for fat tails
                    shock = student_t.rvs(df=8)
                    shock /= np.sqrt(8 / (8 - 2))  # Normalize t-distribution
                    
                    daily_log_return = daily_drift + daily_vol * shock
                    
                    # Add flash crashes occasionally (reduced frequency)
                    if np.random.random() < 0.00015 and current_regime not in [CRASH, CORRECTION]:
                        daily_log_return = np.random.uniform(-0.15, -0.05)
                        current_regime = CRASH
                    
                    # Update price
                    close_prices[j] = close_prices[j-1] * np.exp(daily_log_return)
                    
                    # Generate OHLC
                    daily_range = 0.03 if current_regime in [BEAR, CRASH] else 0.02
                    range_factor = daily_vol / (bull_vol / np.sqrt(252))
                    daily_range *= range_factor
                    
                    # Open price typically between previous close and current close
                    open_frac = np.clip(np.random.normal(0.5, 0.2), 0, 1)
                    open_prices[j] = close_prices[j-1] + (close_prices[j] - close_prices[j-1]) * open_frac
                    
                    # High/Low based on direction of move
                    if close_prices[j] > close_prices[j-1]:
                        high_prices[j] = max(open_prices[j], close_prices[j]) * (1 + np.random.uniform(0, daily_range))
                        low_prices[j] = min(open_prices[j], close_prices[j]) * (1 - np.random.uniform(0, daily_range * 0.5))
                    else:
                        high_prices[j] = max(open_prices[j], close_prices[j]) * (1 + np.random.uniform(0, daily_range * 0.5))
                        low_prices[j] = min(open_prices[j], close_prices[j]) * (1 - np.random.uniform(0, daily_range))
                    
                    # Make sure OHLC relationships are valid
                    high_prices[j] = max(high_prices[j], open_prices[j], close_prices[j])
                    low_prices[j] = min(low_prices[j], open_prices[j], close_prices[j])
                
                # Round price values for realism
                open_prices = np.round(open_prices, 2)
                high_prices = np.round(high_prices, 2)
                low_prices = np.round(low_prices, 2)
                close_prices = np.round(close_prices, 2)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Date': dates,
                    'Open': open_prices,
                    'High': high_prices,
                    'Low': low_prices,
                    'Close': close_prices,
                    'Regime': regimes
                })
                df.set_index('Date', inplace=True)
                
                # Calculate technical indicators using talib
                # Moving Averages
                df['MA5'] = talib.SMA(df['Close'].values, timeperiod=5)
                df['MA20'] = talib.SMA(df['Close'].values, timeperiod=20)
                df['MA50'] = talib.SMA(df['Close'].values, timeperiod=50)
                df['MA200'] = talib.SMA(df['Close'].values, timeperiod=200)
                
                # RSI
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                df['BB_width'] = (upper - lower) / middle
                
                # ATR
                df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                
                # Returns
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                df['Return_1W'] = df['Close'].pct_change(5)
                df['Return_1M'] = df['Close'].pct_change(21)
                df['Return_3M'] = df['Close'].pct_change(63)
                
                # Drawdowns
                rolling_max = df['Close'].cummax()
                df['CurrentDrawdown'] = (df['Close'] / rolling_max) - 1
                
                # Max drawdown over rolling window
                df['MaxDrawdown_252d'] = df['CurrentDrawdown'].rolling(252).min()
                
                # Sharpe ratios
                df['Sharpe_20d'] = (df['LogReturn'].rolling(20).mean() / df['LogReturn'].rolling(20).std()) * np.sqrt(252)
                df['Sharpe_60d'] = (df['LogReturn'].rolling(60).mean() / df['LogReturn'].rolling(60).std()) * np.sqrt(252)
                df['Sharpe_252d'] = (df['LogReturn'].rolling(252).mean() / df['LogReturn'].rolling(252).std()) * np.sqrt(252)
                
                # Drop NaN values (from indicators that need lookback periods)
                df.dropna(inplace=True)
                
                # Check if this data meets our criteria for good US equities
                annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252 / len(df)) - 1
                overall_sharpe = (df['LogReturn'].mean() / df['LogReturn'].std()) * np.sqrt(252)
                
                # Check if data meets criteria
                if overall_sharpe >= min_sharpe and annual_return >= min_annual_return:
                    # Add metadata about performance
                    df.attrs['annualized_return'] = annual_return
                    df.attrs['sharpe_ratio'] = overall_sharpe
                    df.attrs['max_drawdown'] = df['CurrentDrawdown'].min()
                    
                    synthetic_data_list.append(df)
                    # print(f"Generated stock with Sharpe: {overall_sharpe:.2f}, Annual return: {annual_return:.2%}")
                    break
                    
                if attempts == max_attempts:
                    print(f"Warning: Failed to generate a stock meeting criteria after {max_attempts} attempts. Relaxing constraints.")
                    # If we've tried many times, relax the constraints
                    min_sharpe *= 0.8
                    min_annual_return *= 0.8
        
        return synthetic_data_list

    def plot_comparison(self, synthetic_data, num_bars=350):
        if self.original_data is None:
            raise ValueError("Original data must be provided for comparison plotting")
            
        original_slice = self.original_data.iloc[-num_bars:]
        synthetic_slice = synthetic_data.iloc[:min(len(synthetic_data), num_bars)]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        self._plot_candles(axes[0], original_slice, "Original OHLC Data")
        self._plot_candles(axes[1], synthetic_slice, "Synthetic OHLC Data")
        plt.tight_layout()
        return fig

    def _plot_candles(self, ax, data, title):
        for i, row in enumerate(data.itertuples()):
            o, h, l, c = row.open, row.high, row.low, row.close
            color = 'green' if c >= o else 'red'
            bottom = o if c >= o else c
            height = abs(c - o)
            ax.add_patch(plt.Rectangle((i - 0.3, bottom), 0.6, height, color=color, alpha=0.5))
            ax.plot([i, i], [l, h], color='black', linewidth=1)
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.set_xlim(-1, len(data))