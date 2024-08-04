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

    def split(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
              y: Union[np.ndarray, pd.Series, None] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Split the input data into random windows, allowing duplicates if necessary.
        
        Args:
            X (np.ndarray, pd.DataFrame, or pd.Series): Input time series data.
            y (np.ndarray, pd.Series, optional): Target values. If provided, should have the same length as X.
        
        Yields:
            Tuple of train and test indices for each split.
        """
        n_samples = self._get_n_samples(X)
        indices = self._get_indices(X)

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
            test_indices = train_end - 1 + self.fh
            
            yield indices[train_start:train_end], indices[test_indices]

    def get_n_splits(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
                     y: Union[np.ndarray, pd.Series, None] = None) -> int:
        """
        Returns the number of splitting iterations.
        
        Args:
            X (np.ndarray, pd.DataFrame, or pd.Series): Input time series data.
            y (np.ndarray, pd.Series, optional): Target values. Not used, present for API consistency.
        
        Returns:
            int: Number of splits (same as n_splits or max possible unique splits).
        """
        if self.n_splits is None:
            n_samples = self._get_n_samples(X)
            min_start = self.window_length
            max_start = n_samples - self.window_length - self.max_fh + 1
            return max_start - min_start
        return self.n_splits

    def _get_n_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> int:
        """Helper method to get the number of samples in X."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return len(X)
        elif isinstance(X, np.ndarray):
            return X.shape[0] if X.ndim > 1 else len(X)
        else:
            raise ValueError("X must be a numpy array, pandas DataFrame, or pandas Series")

    def _get_indices(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Helper method to get the indices of X."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.index.values
        elif isinstance(X, np.ndarray):
            return np.arange(self._get_n_samples(X))
        else:
            raise ValueError("X must be a numpy array, pandas DataFrame, or pandas Series")