# helpers.py
import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def create_lagged_features(data, window_size):
    """Create lagged features from a time series data.

    Args:
        data (pd.Series): The time series data to transform.
        window_size (int): The size of the sliding window for creating lagged features.

    Returns:
        pd.DataFrame: The dataframe with lagged features as columns.
    """
    features = pd.DataFrame()
    for i in range(window_size):
        features[f'Lag_{i+1}'] = data.shift(i+1)
    return features


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, metric):
    """Train and evaluate a model using cross-validation and test set.

    Args:
        model (sklearn estimator): The model to train and evaluate.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test labels.
        metric (str): The metric to use for model evaluation: either 'accuracy' or 'mse'.

    Returns:
        float: The cross-validation score on the training set.
        float: The test score on the test set.
    """
    if metric == 'accuracy':
        scoring = 'accuracy'
        error_func = accuracy_score
    elif metric == 'mse':
        scoring = 'neg_mean_squared_error'
        error_func = mean_squared_error
    else:
        raise ValueError('Invalid metric')

    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring).mean()
    
# ```python
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# test_score=error_func(y_test,y_pred)
# return cv_score,test_score


def plot_predictions(y_true,y_pred):
    """Plot the true and predicted values on a line chart.
    Args:
    y_true(pd.Series):The true values.
    y_pred(pd.Series):The predicted values.
    """
    plt.figure(figsize=(10.6))
    plt.plot(y_true.index,y_true.values,label='True')
    plt.plot(y_pred.index,y_pred.values,label='Predicted')
    plt.legend()
    plt.title('True vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.show()