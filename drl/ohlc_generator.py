import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class SimpleOHLCGenerator:
    def __init__(self, original_data):
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

    def plot_comparison(self, synthetic_data, num_bars=350):
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

