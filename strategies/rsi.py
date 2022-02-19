import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf


def main():
    # remove warnings from pandas
    pd.options.mode.chained_assignment = None

    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.Symbol.to_list()
    # replace . with -
    tickers = [x.replace('.', '-') for x in tickers]
    # pop 474 and 489 from the list
    tickers.pop(474)
    tickers.pop(489)

    frame = rsi_calc(tickers[1])
    buy, sell = get_signals(frame)

    print(sell)


# Rsi calculation of asset
def rsi_calc(asset):
    # Get the data from Yahoo Finance
    df = yf.download(asset, start='2022-01-01', interval='1h')
    # Take mean of the last 200 days
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    # FInd the price change
    df['Price_Change'] = df['Adj Close'].pct_change()
    # Define the up moves and down moves
    df['Up_Move'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
    df['Down_Move'] = df['Price_Change'].apply(
        lambda x: abs(x) if x < 0 else 0)
    # Calculate the average up move and down move
    df['Avg_Up_Move'] = df['Up_Move'].ewm(span=19).mean()
    df['Avg_Down_Move'] = df['Down_Move'].ewm(span=19).mean()
    df = df.dropna()
    # Calculate the RSI
    df['RS'] = df['Avg_Up_Move'] / df['Avg_Down_Move']
    # df['RSI'] = 100 - (100 / (1 + df['RS']))
    df['RSI']= df['RS'].apply(lambda x: 100 - (100 / (1 + x)))
    # Check if Adj Close is greater than SMA_200 and RSI is less than 30
    df.loc[(df['Adj Close'] > df['SMA_200']) & (df['RSI'] < 30), 'Buy'] = 'Yes'
    df.loc[(df['Adj Close'] < df['SMA_200']) | (df['RSI'] > 30), 'Buy'] = 'No'

    # Plot the RSI
    # df.plot(y=['RSI'])
    # plt.show()
    return df


def get_signals(df):
    buying_dates = []
    selling_dates = []

    for i in range(len(df)):
        buy_col = df['Buy'].iloc[i]
        if 'Yes' in buy_col:
            # Append the signal to the buying_dates
            buying_dates.append(df.iloc[i+1].name)
            for j in range(1, 11):
                if df['RSI'].iloc[i+j] > 40:
                    selling_dates.append(df.iloc[i+j+1].name)
                    break
                elif j == 10:
                    selling_dates.append(df.iloc[i+j+1].name)
    return buying_dates, selling_dates


main()
