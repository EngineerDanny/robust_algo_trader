# core.py
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from .helpers import create_lagged_features, train_and_evaluate_model, plot_predictions

def main(data_path, window_size, sma_period, bb_period, bb_std, task, models, metric):
    """Run the time series forecasting system.

    Args:
        data_path (str): The path to the CSV file containing the forex data.
        window_size (int): The size of the sliding window for creating lagged features.
        sma_period (int): The period for computing the SMA indicator.
        bb_period (int): The period for computing the BB indicator.
        bb_std (int): The standard deviation for computing the BB indicator.
        task (str): The task to perform: either 'classification' or 'regression'.
        models (list): The models to use: either 'all', 'lr', 'lda' or 'qda'.
        metric (str): The metric to use for model evaluation: either 'accuracy' or 'mse'.
    """
    # Load and preprocess the data
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Compute the technical indicators
    data['SMA'] = data['Close'].rolling(sma_period).mean()
    data['Upper_BB'] = data['SMA'] + bb_std * data['Close'].rolling(bb_period).std()
    data['Lower_BB'] = data['SMA'] - bb_std * data['Close'].rolling(bb_period).std()

    # Create the target variables
    data['Return'] = data['Close'].pct_change()
    if task == 'classification':
        data['Direction'] = np.where(data['Return'] > 0, 1, 0)
        target = 'Direction'
    elif task == 'regression':
        target = 'Return'
    else:
        raise ValueError('Invalid task')

    # Drop missing values
    data.dropna(inplace=True)

    # Create lagged features
    features = create_lagged_features(data['Close'], window_size)
    features[target] = data[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features.drop(target, axis=1), features[target], shuffle=False)

    # Train and evaluate different models
    if models == 'all':
        models = ['lr', 'lda', 'qda']
    
    best_model = None
    best_score = None

    for model in models:
        if model == 'lr':
            estimator = LogisticRegression() if task == 'classification' else LinearRegression()
        elif model == 'lda':
            estimator = LinearDiscriminantAnalysis()
        elif model == 'qda':
            estimator = QuadraticDiscriminantAnalysis()
        else:
            raise ValueError('Invalid model')

        cv_score, test_score = train_and_evaluate_model(estimator, X_train, y_train, X_test, y_test, metric)
        
        print(f'Model: {model}')
        print(f'Cross-validation score: {cv_score}')
        print(f'Test score: {test_score}')
        
        if best_score is None or (metric == 'accuracy' and test_score > best_score) or (metric == 'mse' and test_score < best_score):
            best_model = model
            best_score = test_score
    
    print(f'Best model: {best_model}')
    print(f'Best score: {best_score}')

    # Make predictions on the test set
    if best_model == 'lr':
        estimator = LogisticRegression() if task == 'classification' else LinearRegression()
    elif best_model == 'lda':
        estimator = LinearDiscriminantAnalysis()
    elif best_model == 'qda':
        estimator = QuadraticDiscriminantAnalysis()

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    # Plot the results
    plot_predictions(y_test, y_pred)

if __name__ == '__main__':
    main(data_path='data/EURUSD.csv', window_size=5, sma_period=20, bb_period=20, bb_std=2, task='classification', models='all', metric='accuracy')
    