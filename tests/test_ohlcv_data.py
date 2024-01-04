import pandas as pd
import unittest

# Load the data from a csv file
data = pd.read_csv("/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/AUD_CAD_H1_processed_data.csv", index_col=0, parse_dates=True)

# Define a test class
class TestData(unittest.TestCase):
    def test_data_values(self):
        # Check that the data values are not None
        self.assertTrue((data != None).all().all())
        # Check the the Open, High, Low, Close and Volume columns values are greater than 0
        self.assertTrue((data["Open"] > 0).all())
        self.assertTrue((data["High"] > 0).all())
        self.assertTrue((data["Low"] > 0).all())
        self.assertTrue((data["Close"] > 0).all())
        self.assertTrue((data["Volume"] > 0).all())

    def test_data_time(self):
        # Check that the time is increasing
        # the time column is the index
        self.assertTrue((data.index == data.index.sort_values()).all())
        # Check the format of the time whether it is in the correct format
        self.assertTrue((data.index == pd.to_datetime(data.index)).all())
        

# Run the tests
if __name__ == "__main__":
    unittest.main()
