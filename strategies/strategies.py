from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from ta import momentum
import pandas as pd


def RSI_Indicator(arr, n):
    arr_series = pd.Series(arr)
    rsi_indicator = momentum.RSIIndicator(arr_series, window=n)
    return rsi_indicator.rsi()

class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    
    def init(self):
        self.ma1 = self.I(SMA, self.data.Close, self.n1)
        self.ma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        current_price = self.data.Close[-1]
        if crossover(self.ma1, self.ma2):
            self.buy( sl=current_price * 0.95)
        elif crossover(self.ma2, self.ma1):
            self.sell( sl=current_price * 1.05)
            

class RsiStrategy(Strategy):
    n1 = 30
    n2 = 70
    
    def init(self):
        # Load the price and RSI
        price = self.data.Close
        self.rsi = self.I(RSI_Indicator, price, 14)

    def next(self):
        # Buy when RSI crosses above n1
        if crossover(self.rsi, self.n1):
            self.buy()
        # Sell when RSI crosses below n2
        elif crossover(self.n2, self.rsi):
            self.sell()

# This is a custom strategy class.
# The dataframe passed has a column called 'signal' which is either 1 for buy, -1 for sell or 0 for close if there is an existing trade.
class CustomStrategy(Strategy):
    def init(self):
        # Load the signal from the dataframe
        self.signal = self.data.Signal
    
    def next(self):
        # If there is no existing trade, then take the trade
        current_signal = self.signal[-1]
        if self.position.size == 0:
            if current_signal == 1:
                self.buy()
            elif current_signal == -1:
                self.sell()
        # If there is an existing trade, then close it if the signal is 0
        elif self.position.size > 0:
            if current_signal == -1:
                self.position.close()
        elif self.position.size < 0:
            if current_signal == 1:
                self.position.close()