# test_helpers.py
import unittest

import pandas as pd
import numpy as np

from main import helpers

class TestHelpers(unittest.TestCase):

    def setUp(self):
        # Set up some sample data and parameters for testing
        self.data = pd.Series([1, 2, 3, 4, 5])
        self.window_size = 3
        self.model = LinearRegression()
        self.X_train = pd.DataFrame({'Lag_1': [1, 2], 'Lag_2': [np.nan, 1], 'Lag_3': [np.nan, np.nan]})
        self.y_train = pd.Series([2, 3])
        self.X_test = pd.DataFrame({'Lag_1': [3, 4], 'Lag_2': [2, 3], 'Lag_3': [1, 2]})
        self.y_test = pd.Series([4, 5])
        self.metric = 'mse'

    def test_create_lagged_features(self):
        # Test the create_lagged_features function with the sample data and parameters
        features = helpers.create_lagged_features(self.data, self.window_size)
        # Assert that the features is a dataframe with four columns
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], 4)
        # Assert that the features has the expected column names and values
        expected_columns = ['Lag_1', 'Lag_2', 'Lag_3', 0]
        expected_values = [[np.nan, np.nan, np.nan, 1], [1, np.nan, np.nan, 2], [2, 1, np.nan, 3], [3, 2, 1, 4], [4, 3, 2, 5]]
        self.assertListEqual(list(features.columns), expected_columns)
        self.assertListEqual(list(features.values), expected_values)

    def test_train_and_evaluate_model(self):
        # Test the train_and_evaluate_model function with the sample data and parameters
        cv_score, test_score = helpers.train_and_evaluate_model(self.model, self.X_train, self.y_train, self.X_test, self.y_test, self.metric)
        # Assert that the cv_score and test_score are floats
        self.assertIsInstance(cv_score, float)
        self.assertIsInstance(test_score, float)
        # Assert that the cv_score and test_score are non-negative
        self.assertGreaterEqual(cv_score, 0)
        self.assertGreaterEqual(test_score, 0)

    def test_plot_predictions(self):
        # Test the plot_predictions function with the sample data and parameters
        with patch('robust_algo_trader.helpers.plt.show') as mock_show:
            # Mock the plt.show function to avoid displaying the plot
            helpers.plot_predictions(self.y_test, self.y_test)
            # Assert that the plt.show function was called once
            mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()