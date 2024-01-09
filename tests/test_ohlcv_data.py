import pandas as pd
import unittest

# Load the df from a csv file
df = pd.read_csv("/projects/genomic-ml/da2343/ml_project_2/data/gen_oanda_data/AUD_CAD_H1_processed_data.csv", index_col=0, parse_dates=True)

class TestData(unittest.TestCase):
    def test_data_values(self):
        # Check that the df values are not None
        self.assertTrue((df != None).all().all())
        # Check the the Open, High, Low, Close and Volume columns values are greater than 0
        self.assertTrue((df["Open"] > 0).all())
        self.assertTrue((df["High"] > 0).all())
        self.assertTrue((df["Low"] > 0).all())
        self.assertTrue((df["Close"] > 0).all())
        self.assertTrue((df["Volume"] > 0).all())

    def test_data_time(self):
        # Check that the time is increasing
        # the time column is the index
        self.assertTrue((df.index == df.index.sort_values()).all())
        # Check the format of the time whether it is in the correct format
        self.assertTrue((df.index == pd.to_datetime(df.index)).all())
    
    def test_macd(self):
        # Check that the MACD_Crossover_Change column has only 2, 0, -2 values
        self.assertTrue((df["MACD_Crossover_Change"].isin([2, 0, -2])).all())

# Run the tests
if __name__ == "__main__":
    unittest.main()
