import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from numba import jit


class RandomStartSlidingWindowSplitter(TimeSeriesSplit):
    """Time Series cross-validator with random starting points for each split.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate. This cross-validation object is a variation
    of TimeSeriesSplit with the following differences:
    * The starting point of each split is randomized within a certain range.
    * The range of possible starting points is controlled by the 'randomness' parameter.
    * Both train_size and test_size can be specified.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    
    train_size : int, default=None
        Number of samples in the training set. If None, it will be set to the maximum
        possible size that allows for the specified test_size in all splits.
    
    test_size : int, default=None
        Number of samples in the test set. If None, it will default to
        n_samples // (n_splits + 1).
    
    randomness : float, default=0.2
        A value between 0 and 1 that determines the range of possible random starting points.
        0 means no randomness (equivalent to regular TimeSeriesSplit),
        1 means maximum randomness (can start anywhere before the last possible regular split).
    """
    def __init__(self, n_splits=5, train_size=None, test_size=None, randomness=0.2):
        super().__init__(n_splits=n_splits)
        self.train_size = train_size
        self.test_size = test_size
        if not 0 <= randomness <= 1:
            raise ValueError("randomness must be between 0 and 1")
        self.randomness = randomness

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_splits = self.n_splits
        
        # Determine test_size if not specified
        if self.test_size is None:
            test_size = n_samples // (n_splits + 1)
        else:
            test_size = self.test_size
        
        # Determine train_size if not specified
        if self.train_size is None:
            train_size = n_samples - (test_size * n_splits)
        else:
            train_size = self.train_size
        
        if train_size + test_size > n_samples:
            raise ValueError(
                f"Cannot have train_size={train_size} and test_size={test_size} "
                f"with n_samples={n_samples}")

        # The last possible start index for a regular TimeSeriesSplit
        last_regular_start = n_samples - (n_splits * test_size + train_size)
        
        # Calculate the range of possible random starts
        random_range = int(last_regular_start * self.randomness)
        
        # Generate random starting points
        random_starts = np.random.randint(0, random_range + 1, size=n_splits)
        
        for i in range(n_splits):
            start = random_starts[i]
            train_end = start + train_size
            test_end = train_end + test_size

            if test_end > n_samples:
                test_end = n_samples

            yield np.arange(start, train_end), np.arange(train_end, test_end)
            
            
            
            
