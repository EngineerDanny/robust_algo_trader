import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import talib
import json
import warnings
import matplotlib.pyplot as plt
import json
import time
import datetime as dt
import requests
import re
import itertools
import talib
import logging
from decimal import Decimal
import io

warnings.filterwarnings("ignore")
params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

param_dict = dict(params_df.iloc[param_row, :])
dataset_name = param_dict["dataset_name"]
strategy = param_dict["strategy"]
atr_delta = param_dict["atr_delta"]

# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
    config = json.load(f)

config_settings = config["trading_settings"][dataset_name]
start_hr = config_settings["start_hour"]
end_hr = config_settings["end_hour"]

root_data_dir = config["paths"]["oanda_dir"]
