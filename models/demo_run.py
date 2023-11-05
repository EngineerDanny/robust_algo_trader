import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
import pickle
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.neighbors import *


warnings.filterwarnings('ignore')

params_df = pd.read_csv("params.csv")
if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
  config = json.load(f)
  
param_dict = dict(params_df.iloc[param_row, :])
symbol = param_dict["symbol"]
model_version = config["model_version"]  
  
root_symbol_dir = f"/projects/genomic-ml/da2343/ml_project_2/data/{symbol}"
root_model_dir = f"/projects/genomic-ml/da2343/ml_project_2/models/{model_version}"
synthetic_data = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_SYNTHETIC.csv"

trades_df = pd.read_csv(synthetic_data)
# drop index column
trades_df = trades_df.drop(columns=['index'])

X_train, y_train = trades_df.iloc[:, :-1].to_numpy(), trades_df.iloc[:, -1]
model = LogisticRegressionCV(n_jobs=-1)
model.fit(X_train, y_train)

# save the model to file
if not os.path.exists(root_model_dir):
    os.makedirs(root_model_dir)


model_path = f"{root_model_dir}/{symbol.lower()}.pkl"
pickle.dump(model, open(model_path, "wb"))