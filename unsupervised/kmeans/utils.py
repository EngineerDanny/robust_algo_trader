import pandas as pd
import numpy as np
from typing import Tuple, Iterator, Optional

class OrderedSlidingWindowSplitter:
    def __init__(self, train_weeks: int, test_weeks: int = 2, step_size: int = 1, 
                 allow_partial_window: bool = True, min_test_size: float = 0.85):
        """
        Initialize the OrderedSlidingWindowSplitter.

        Args:
            train_weeks (int): Number of weeks for the training data.
            test_weeks (int): Number of weeks for the test data.
            step_size (int): Number of weeks to slide the window.
            allow_partial_window (bool): Whether to allow partial windows at the end of the dataset.
            min_test_size (float): Minimum size of test set as a fraction of expected size.
        """
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.step_size = step_size
        self.allow_partial_window = allow_partial_window
        self.min_test_size = min_test_size

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate sliding window splits.

        Args:
            X (pd.DataFrame): Input data with DatetimeIndex.

        Yields:
            Tuple containing:
                - train indices
                - test indices
        """
        self._validate_input(X)
        start_date = X.index[0]
        end_date = X.index[-1]

        expected_train_points = self.train_weeks * 5 * 24 * 4
        expected_test_points = self.test_weeks * 5 * 24 * 4
        min_test_points = int(expected_test_points * self.min_test_size)

        window_count = 0
        while start_date + pd.Timedelta(weeks=self.train_weeks + self.test_weeks) <= end_date:
            train_start = self._next_sunday_open(start_date)
            train_end = self._friday_close_after_weeks(train_start, self.train_weeks)
            test_start = self._next_sunday_open(train_end)
            test_end = self._friday_close_after_weeks(test_start, self.test_weeks)

            train_mask = self._create_market_hours_mask(X, train_start, train_end)
            test_mask = self._create_market_hours_mask(X, test_start, min(test_end, end_date))

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(test_indices) < min_test_points and not self.allow_partial_window:
                break

            yield train_indices, test_indices

            start_date += pd.Timedelta(weeks=self.step_size)
            window_count += 1

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """
        Calculate the number of splits.

        Args:
            X (pd.DataFrame): Input data with DatetimeIndex.

        Returns:
            int: Number of splits.
        """
        self._validate_input(X)
        return sum(1 for _ in self.split(X))

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate the input data.

        Args:
            X (pd.DataFrame): Input data with DatetimeIndex.

        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Input data must have a DatetimeIndex")
        if len(X) < (self.train_weeks + self.test_weeks) * 5 * 24 * 4:  # Assuming 15-minute intervals, 5 days a week
            raise ValueError("Insufficient data for at least one split")

    def _next_sunday_open(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Find the next Sunday 22:00.

        Args:
            date (pd.Timestamp): Starting date.

        Returns:
            pd.Timestamp: Next Sunday at 22:00.
        """
        next_sunday = date + pd.Timedelta(days=(6 - date.dayofweek) % 7)
        return next_sunday.replace(hour=22, minute=0, second=0, microsecond=0)

    def _friday_close_after_weeks(self, start: pd.Timestamp, weeks: int) -> pd.Timestamp:
        """
        Find the Friday 21:45 after the specified number of weeks.

        Args:
            start (pd.Timestamp): Starting date.
            weeks (int): Number of weeks to add.

        Returns:
            pd.Timestamp: Friday at 21:45 after the specified number of weeks.
        """
        end = start + pd.Timedelta(weeks=weeks, days=-2, hours=21, minutes=45)
        return end

    def _create_market_hours_mask(self, X: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """
        Create a boolean mask for market hours within the given period.

        Args:
            X (pd.DataFrame): Input data with DatetimeIndex.
            start (pd.Timestamp): Start of the period.
            end (pd.Timestamp): End of the period.

        Returns:
            np.ndarray: Boolean mask for market hours.
        """
        mask = (X.index >= start) & (X.index <= end)
        mask &= ((X.index.dayofweek < 5) | ((X.index.dayofweek == 6) & (X.index.hour >= 22)))
        return mask

    def plot_splits(self, X: pd.DataFrame, n_splits: Optional[int] = None) -> None:
        """
        Plot the splits for visualization.

        Args:
            X (pd.DataFrame): Input data with DatetimeIndex.
            n_splits (int, optional): Number of splits to plot. If None, plot all splits.
        """
        import matplotlib.pyplot as plt

        splits = list(self.split(X))
        if n_splits is not None:
            splits = splits[:n_splits]

        fig, ax = plt.subplots(figsize=(15, 5 * len(splits)))
        for i, (train_idx, test_idx) in enumerate(splits):
            ax.plot(X.index[train_idx], [i] * len(train_idx), 'b.', label='Train' if i == 0 else '')
            ax.plot(X.index[test_idx], [i] * len(test_idx), 'r.', label='Test' if i == 0 else '')

        ax.set_yticks(range(len(splits)))
        ax.set_yticklabels([f'Split {i+1}' for i in range(len(splits))])
        ax.legend()
        plt.title('Ordered Sliding Window Splits')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()