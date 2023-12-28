import os
import json 

config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
# config_path = "/Users/newuser/Projects/robust-algo-trader/settings/config.json"
saved_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data/saved_data"
results_dir = "/projects/genomic-ml/da2343/ml_project_2/data/gen_trade_data/results"
with open(config_path) as f:
  config = json.load(f) 
config_settings = config["trading_settings"]

# create train_zero and train_one directories
train_zero_dir = os.path.join(results_dir, "0")
train_one_dir = os.path.join(results_dir, "1")
if not os.path.exists(train_zero_dir):
    os.makedirs(train_zero_dir)
if not os.path.exists(train_one_dir):
    os.makedirs(train_one_dir)

for key, value in config_settings.items():
    label_zero_dir = f"{saved_data_dir}/{key}/0"
    label_one_dir = f"{saved_data_dir}/{key}/1"
    os.system(f"cp {label_zero_dir}/* {train_zero_dir}")
    os.system(f"cp {label_one_dir}/* {train_one_dir}")
    print(f"copied files from {label_zero_dir} to {train_zero_dir}")