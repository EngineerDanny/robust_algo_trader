import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
from sklearn.linear_model import *
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.model_selection import SlidingWindowSplitter
import warnings
import pandas as pd
import numpy as np
import plotnine as p9
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.pipeline import *

from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.model_selection import *
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib
import plotnine as p9

warnings.filterwarnings('ignore')

params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

param_dict = dict(params_df.iloc[param_row, :])

dataset_name = param_dict["dataset_name"]
algorithm = param_dict["algorithm"]
param_fold_id = param_dict["fold_id"]

root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data" 

dataset_dict = {"EURUSD_H1_2011_2015_TRADES_binary" : f"{root_data_dir}/EURUSD_H1_2011_2015_TRADES_binary.tsv"}
dataset_path = dataset_dict[dataset_name]


classifier_dict = {
    "KNeighborsClassifier": GridSearchCV(KNeighborsClassifier(),
                                            param_grid={'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                                            cv=5,
                                            scoring='accuracy',
                                            verbose=0,
                                            n_jobs=-1),
    "SVC": GridSearchCV(SVC(),
                        param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
                        cv=5,
                        scoring='accuracy',
                        verbose=0,
                        n_jobs=-1),
    "DecisionTreeClassifier": GridSearchCV(DecisionTreeClassifier(),
                                            param_grid={'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                                            cv=5,
                                            scoring='accuracy',
                                            verbose=0,
                                            n_jobs=-1),
    "MLPClassifier" : GridSearchCV(MLPClassifier(),
                                    param_grid={'hidden_layer_sizes': [(50,50,50), (50,100,50), (100, 300, 200, 100), (100,)],
                                                'activation': ['tanh', 'relu', 'logistic'],
                                                'solver': ['sgd', 'adam', 'lbfgs'],
                                                'alpha': [0.0001, 0.001, 0.01, 0.1],
                                                'learning_rate': ['constant','adaptive'],
                                                },
                                    cv=5,
                                    scoring='accuracy',
                                    verbose=0,
                                    n_jobs=-1),
    
    "AdaBoostClassifier": GridSearchCV(AdaBoostClassifier(),
                                        param_grid={'n_estimators': [50, 100, 200, 500],
                                                    'learning_rate': [0.1, 0.5, 1.0, 2.0]},
                                        cv=5,
                                        scoring='accuracy',
                                        verbose=0,
                                        n_jobs=-1),
    "RandomForestClassifier": GridSearchCV(RandomForestClassifier(),
                                            param_grid={'n_estimators': [50, 100, 200, 500],
                                                        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                                            cv=5,
                                            scoring='accuracy',
                                            verbose=0,
                                            n_jobs=-1),
    'DummyClassifier': DummyClassifier(strategy="most_frequent"),
    'LogisticRegressionCV': LogisticRegressionCV(cv=5, 
                                                 random_state=0, 
                                                 max_iter=100_000, 
                                                 n_jobs=-1),
    
}


for data_set, (input_mat, output_vec) in data_dict.items():
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold_id, (train_index, test_index) in enumerate(kf.split(input_mat)):
        if param_fold_id != fold_id:
            continue
        X_train, X_test = input_mat[train_index], input_mat[test_index]
        y_train, y_test = output_vec[train_index], output_vec[test_index]

        pred_dict = {}
        # iterate over classifiers
        for name, clf in classifier_dict.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pred_dict[name] = y_pred
        
        for algorithm, y_pred in pred_dict.items():
            test_acc_dict = {
                "test_accuracy_percent": accuracy_score(y_test, y_pred) * 100,
                # "precision_score": precision_score(y_test, y_pred),
                # "f1_score": f1_score(y_test, y_pred),
                "data_set": dataset_name,
                "fold_id": fold_id,
                "algorithm": algorithm
            }
            test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))
test_acc_df = pd.concat(test_acc_df_list)

# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
test_err_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")
