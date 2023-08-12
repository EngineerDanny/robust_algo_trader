import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import json


warnings.filterwarnings('ignore')

params_df = pd.read_csv("params.csv")
if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

param_dict = dict(params_df.iloc[param_row, :])
symbol = param_dict["symbol"]

with open(config_path) as f:
  config = json.load(f)
  
model_version = config["model_version"]  
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
root_symbol_dir = f"/projects/genomic-ml/da2343/ml_project_2/data/{symbol}"
root_model_dir = f"/projects/genomic-ml/da2343/ml_project_2/models/{symbol}"

real_data = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_REAL.csv"
synthetic_data = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_SYNTHETIC.csv"


generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)