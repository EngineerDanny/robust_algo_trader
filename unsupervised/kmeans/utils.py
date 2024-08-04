import numpy as np
import pandas as pd
from typing import Tuple, Union, Iterator

class RandomStartWindowSplitter:
    def __init__(self, window_length: int, fh: Union[int, list, np.ndarray], 
                 n_splits: int = None, random_state: int = None):
        """
        Initialize the RandomStartWindowSplitter.
        
        Args:
            window_length (int): The size of each window.
            fh (int, list, or np.ndarray): Forecasting horizon, relative time points to forecast.
            n_splits (int, optional): Number of random splits to generate. If None, uses max possible unique splits.
            random_state (int, optional): Seed for random number generator.
        """
        if window_length <= 0:
            raise ValueError("window_length must be a positive integer")
        self.window_length = window_length
        
        if isinstance(fh, (int, list, np.ndarray)):
            self.fh = np.array([fh] if isinstance(fh, int) else fh)
        else:
            raise ValueError("fh must be an int, list, or numpy array")
        
        if len(self.fh) == 0:
            raise ValueError("fh must not be empty")
        
        self.min_fh = min(self.fh)
        self.max_fh = max(self.fh)
        
        if self.min_fh <= 0:
            raise ValueError("All values in fh must be positive")
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Split the input data into random windows, allowing duplicates if necessary.
        
        Args:
            X (pd.DataFrame): Input time series data with DatetimeIndex.
        
        Yields:
            Tuple of train and test integer indices for each split.
        """
        n_samples = len(X)

        if n_samples < self.window_length + self.max_fh:
            raise ValueError(f"Insufficient data: n_samples ({n_samples}) must be at least window_length ({self.window_length}) + max(fh) ({self.max_fh})")

        min_start = self.window_length
        max_start = n_samples - self.window_length - self.max_fh + 1
        available_starts = max_start - min_start

        if self.n_splits is None:
            self.n_splits = available_starts

        # Generate all possible start indices and shuffle them
        all_start_indices = np.arange(min_start, max_start)
        self.rng.shuffle(all_start_indices)
        
        # Use modulo to allow for wrapping around when n_splits > available_starts
        for i in range(self.n_splits):
            start_idx = all_start_indices[i % available_starts]
            train_start = start_idx
            train_end = train_start + self.window_length
            test_end = train_end + self.max_fh
            
            yield np.arange(train_start, train_end), np.arange(train_end, test_end)

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """
        Returns the number of splitting iterations.
        
        Args:
            X (pd.DataFrame): Input time series data with DatetimeIndex.
        
        Returns:
            int: Number of splits (same as n_splits or max possible unique splits).
        """
        if self.n_splits is None:
            n_samples = len(X)
            min_start = self.window_length
            max_start = n_samples - self.window_length - self.max_fh + 1
            return max_start - min_start
        return self.n_splits