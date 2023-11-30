# Robust Algo Trader

- Robust Algo Trader is a Python-based project that aims to create a statistically robust process for trading financial time series markets using machine learning techniques. 

- The project focuses on forex data, which consists of open, high, low and close (OHLC) prices, as well as technical indicators such as simple moving average (SMA) and Bollinger bands (BB).

## Experiment 1 - Using ML to forecast close prices 
**Aproach**: Train ML algorithms on historic data (close prices) to forecast future close prices.

**Algos Used**: Exponential Smoothing, Bayesian Ridge, BlockRNNModel, etc

**Test**: Test forecast prediction with actual labels and find the MAPE between prediction and labels.

**Problem**: Too much noise in the data so ML algos cannot learn the signal, algos learn the noise instead.

![exp1](https://github.com/EngineerDanny/robust_algo_trader/blob/main/assets/experiments/exp1.jpeg)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
