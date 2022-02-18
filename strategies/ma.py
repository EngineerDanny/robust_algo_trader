import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as web


# Moving Averages
ma_1 = 5
ma_2 = 10
# No of years to plot
yr = 1

# Set the start and end dates
start = dt.datetime.now() - dt.timedelta(days=365*yr)
end = dt.datetime.now()

# Get the data from Yahoo Finance
# time frame is 1 hour
data = web.DataReader('AAPL', 'yahoo-dividends', start, end, retry_count=5,interval = 'm')
web.data.read

adj_close = data['Adj Close']
data[f'MA_{ma_1}'] = adj_close.rolling(ma_1).mean()
data[f'MA_{ma_2}'] = adj_close.rolling(ma_2).mean()

data = data.iloc[ma_2:]

# Buy when the moving average crosses the other
buy_signals = []
sell_signals = []
trigger = 0

for x in range(len(data)):
    if data[f'MA_{ma_1}'].iloc[x] > data[f'MA_{ma_2}'].iloc[x] and trigger != 1:
        buy_signals.append(data['Adj Close'].iloc[x])
        sell_signals.append(float('nan'))
        trigger = 1
    elif data[f'MA_{ma_1}'].iloc[x] < data[f'MA_{ma_2}'].iloc[x] and trigger != -1:
        sell_signals.append(data['Adj Close'].iloc[x])
        buy_signals.append(float('nan'))
        trigger = -1
    else:
        buy_signals.append(float('nan'))
        sell_signals.append(float('nan'))

data['Buy'] = buy_signals
data['Sell'] = sell_signals

# set the backgroud color of the plot to dark
plt.style.use('dark_background')
plt.plot(data.index, data['Adj Close'], label='Share Price', alpha=0.5)
plt.plot(data.index, data[f'MA_{ma_1}'],
         label=f'MA_{ma_1}', color='pink', linestyle="--")
plt.plot(data.index, data[f'MA_{ma_2}'],
         label=f'MA_{ma_2}', color='orange', linestyle="--")

# Plot the buy and sell signals
plt.scatter(data.loc[data['Buy'] != float('nan')].index, data['Buy'], c='green', s=100, label='Buy', marker='^', lw = 3)
plt.scatter(data.loc[data['Sell'] != float('nan')].index, data['Sell'], c='red', s=100, label='Sell',marker='v', lw = 3)    

plt.legend(loc='upper left')
plt.show()

print(data)
