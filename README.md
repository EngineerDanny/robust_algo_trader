# Robust Algo Trader

Robust Algo Trader is a Python-based project that aims to create a statistically robust process for forecasting financial time series using machine learning techniques. The project focuses on forex data, which consists of open, high, low and close (OHLC) prices, as well as technical indicators such as simple moving average (SMA) and Bollinger bands (BB).

The project uses supervised learning methods such as logistic regression, linear discriminant analysis and quadratic discriminant analysis to perform both classification and regression tasks on the time series data. The goal is to predict the market direction (up or down) and the magnitude of the return for a given day.

The project leverages scikit-learn, a machine learning library for Python, which provides implementations of many machine learning algorithms and tools for data preprocessing, model evaluation and hyperparameter tuning. The project also uses pandas and numpy for data manipulation, matplotlib and seaborn for data visualization, and statsmodels for statistical analysis.

## Installation

To install the project, you need to have Python 3.10 or higher and pip installed on your system. You can then clone this repository using the following command:

```bash
git clone https://github.com/your_username/robust-algo-trader.git
```

Then, navigate to the project directory and install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the project, you need to have some forex data in CSV format. You can download some sample data from [here](https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1/). The data should have the following columns: Date, Open, High, Low, Close.

You can then run the main.py script using the following command:

```bash
python main.py --data_path path/to/your/data.csv
```

The script will perform the following steps:

- Load and preprocess the data
- Compute the technical indicators (SMA and BB)
- Create the target variables (market direction and return)
- Split the data into train and test sets
- Train and evaluate different models using cross-validation
- Select the best model based on accuracy or mean squared error
- Make predictions on the test set and plot the results

You can also use some optional arguments to customize the script:

- `--window_size`: The size of the sliding window for creating lagged features. Default is 5.
- `--sma_period`: The period for computing the SMA indicator. Default is 20.
- `--bb_period`: The period for computing the BB indicator. Default is 20.
- `--bb_std`: The standard deviation for computing the BB indicator. Default is 2.
- `--task`: The task to perform: either 'classification' or 'regression'. Default is 'classification'.
- `--models`: The models to use: either 'all', 'lr', 'lda' or 'qda'. Default is 'all'.
- `--metric`: The metric to use for model selection: either 'accuracy' or 'mse'. Default is 'accuracy'.

For example, you can run the script with a window size of 10, a SMA period of 50, a BB period of 50, a BB standard deviation of 3, a regression task, and only logistic regression model using the following command:

```bash
python main.py --data_path path/to/your/data.csv --window_size 10 --sma_period 50 --bb_period 50 --bb_std 3 --task regression --models lr --metric mse
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.