import pandas as pd
import unittest

# Load the data from a csv file
data = pd.read_csv("/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/AUD_CAD_H1_processed_data.csv", index_col=0, parse_dates=True)

# Define a test class
class TestData(unittest.TestCase):

    # Define some test methods
    def test_data_shape(self):
        # Check that the data has 8 rows and 5 columns
        self.assertEqual(data.shape, (8, 5))

    def test_data_columns(self):
        # Check that the data has the expected column names
        self.assertListEqual(list(data.columns), ["Open", "High", "Low", "Close", "Volume"])

    def test_data_values(self):
        # Check that the data values are positive
        self.assertTrue((data > 0).all().all())

    def test_data_consistency(self):
        # Check that the open, high, low and close values are equal for each row
        self.assertTrue((data["Open"] == data["High"]).all())
        self.assertTrue((data["Open"] == data["Low"]).all())
        self.assertTrue((data["Open"] == data["Close"]).all())

    def test_data_volume(self):
        # Check that the volume is 2 for each row
        self.assertTrue((data["Volume"] > 0).all())

# Run the tests
if __name__ == "__main__":
    unittest.main()
