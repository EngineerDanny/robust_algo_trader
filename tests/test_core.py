# test_core.py
import unittest
from unittest.mock import patch

from main import core

class TestMain(unittest.TestCase):

    def setUp(self):
        # Set up some sample data and parameters for testing
        self.data_path = 'sample_data.csv'
        self.window_size = 5
        self.sma_period = 10
        self.bb_period = 10
        self.bb_std = 2
        self.task = 'classification'
        self.models = ['lr']
        self.metric = 'accuracy'

    def test_main(self):
        # Test the main function with the sample data and parameters
        with patch('robust_algo_trader.core.plot_predictions') as mock_plot:
            # Mock the plot_predictions function to avoid displaying the plot
            core.main(self.data_path, self.window_size, self.sma_period, self.bb_period, self.bb_std, self.task, self.models, self.metric)
            # Assert that the plot_predictions function was called once
            mock_plot.assert_called_once()
            # Assert that the best model was logistic regression
            self.assertEqual(core.best_model, 'lr')
            # Assert that the best score was greater than 0.5
            self.assertGreater(core.best_score, 0.5)

if __name__ == '__main__':
    unittest.main()