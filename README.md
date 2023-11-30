# Robust Algo Trader

- Robust Algo Trader is a Python-based project that aims to create a statistically robust process for trading financial time series markets using machine learning techniques. 

- The project focuses on forex data, which consists of open, high, low and close (OHLC) prices, as well as technical indicators such as simple moving average (SMA) and Bollinger bands (BB).

## [Experiment 1 - Using ML to Forecast Close Prices](https://github.com/EngineerDanny/robust_algo_trader/tree/main/forecast)
**Aproach**: Train ML algorithms on historic data (close prices) to forecast future close prices.

**Algos Used**: 
- Theta
- Exponential Smoothing
- Linear Regression
- Bayesian Ridge
- BlockRNNModel

**Test**: Test forecast prediction with actual labels and find the MAPE between prediction and labels.

**Problem**: 
- Too much noise in the data so ML algos cannot learn the signal, algos learn the noise instead.
- Continuous Retraining of algos using historic data from window size takes time.
- It is not stable and maintable.

![exp1](https://github.com/EngineerDanny/robust_algo_trader/blob/main/assets/experiments/exp1.jpeg)


## [Experiment 2 - Using ML to Forecast Simple Moving Average (SMA) of Close Prices](https://github.com/EngineerDanny/robust_algo_trader/tree/main/hpc/forecast_tune)
**Aproach**: Remove the noise by using SMA as the label instead of close prices. Train ML algorithms on historic data (SMA) to forecast future SMA.

**Algos Used**:
- Linear Regression
- LassoCV
- RidgeCV

**Test**: Test forecast prediction with actual labels and find the MAPE between prediction and labels.

**Achievement**: Forecast is more accurate and robust than Experiment 1 especially when the window size is larger.

**Problem**:
- The forecast is not accurate and robust enough to be used for trading.
- SMA is a lagging indicator, it is not a good label to use for forecasting.
- Signal trend is not clearly defined and sometimes not captured by the forecast.
- It is not stable and maintable.

![Screenshot 2023-11-29 at 5 48 28 PM](https://github.com/EngineerDanny/robust_algo_trader/assets/47421661/40c368de-1f77-494a-9200-4d392d6debfc)


## [Experiment 3 - MACD Crossover Entry Point, Get Features and Predict the label](https://github.com/EngineerDanny/robust_algo_trader/tree/main/hpc/forecast_tune)

**Aproach**: 
- Use MACD crossover as the entry point to get features and predict the Label.
- If the MACD crossover is positive, the trade is long, otherwise the trade is short.
- The label describes whether the trade is profitable or not (1 or 0).
- The features are only technical indicators with well defined range (RSI, ADX etc).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
