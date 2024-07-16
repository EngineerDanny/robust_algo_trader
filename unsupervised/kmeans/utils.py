import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class OverlappingRandomStartSlidingWindowSplitter(TimeSeriesSplit):
    """Time Series cross-validator with random starting points for each split, allowing overlaps.
    
    This cross-validation object is a variation of TimeSeriesSplit with the following differences:
    * The starting point of each split is randomized within the available range.
    * Splits are allowed to overlap.
    * Both train_size and test_size can be specified.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    
    train_size : int, default=None
        Number of samples in the training set.
    
    test_size : int, default=None
        Number of samples in the test set.
    
    randomness : float, default=1.0
        A value between 0 and 1 that determines the range of possible random starting points.
        0 means evenly spaced splits, 1 means maximum randomness.
    """
    def __init__(self, n_splits=5, train_size=None, test_size=None, randomness=1.0):
        super().__init__(n_splits=n_splits)
        self.train_size = train_size
        self.test_size = test_size
        if not 0 <= randomness <= 1:
            raise ValueError("randomness must be between 0 and 1")
        self.randomness = randomness

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_splits = self.n_splits
        
        if self.train_size is None or self.test_size is None:
            raise ValueError("Both train_size and test_size must be specified")
        
        train_size = self.train_size
        test_size = self.test_size
        
        if train_size + test_size > n_samples:
            raise ValueError(
                f"train_size ({train_size}) + test_size ({test_size}) "
                f"should be <= n_samples ({n_samples})")

        # Calculate the range for start indices
        max_start = n_samples - (train_size + test_size)
        
        if max_start < 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for the specified "
                f"train_size ({train_size}) and test_size ({test_size})")

        # Generate random starting points
        if self.randomness == 0:
            # Evenly spaced splits
            starts = np.linspace(0, max_start, n_splits, dtype=int)
        else:
            # Random starts
            starts = np.random.randint(0, max_start + 1, size=n_splits)
            starts.sort()  # Ensure chronological order

        for start in starts:
            train_end = start + train_size
            test_end = min(train_end + test_size, n_samples)
            
            # Adjust if the test set goes beyond n_samples
            if test_end > n_samples:
                test_end = n_samples
                train_end = test_end - test_size
                start = train_end - train_size

            yield np.arange(start, train_end), np.arange(train_end, test_end)