# Robust Algo Trader

- Robust Algo Trader is a Python-based project that aims to create a statistically robust process for trading financial time series markets using machine learning techniques. 

- The project focuses on forex data, which consists of open, high, low and close (OHLC) prices, as well as technical indicators such as simple moving average (SMA) and Bollinger bands (BB).

## [Experiment 1 - Using ML to forecast close prices](https://github.com/EngineerDanny/robust_algo_trader/tree/main/forecast)
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


## [Experiment 2 - Using ML to forecast Simple Moving Average (SMA)](https://github.com/EngineerDanny/robust_algo_trader/tree/main/hpc/forecast_tune)
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



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
