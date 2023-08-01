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


# input dataset
root_symbol_dir = f"/projects/genomic-ml/da2343/ml_project_2/data/{symbol}"
input_data = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_REAL.csv"

# location of two output files
synthetic_data = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_SYNTHETIC.csv"
description_file = f"{root_symbol_dir}/{symbol}_H1_2011_2015_TRADES_SYNTHETIC_DESCRIPTION.json"


# An attribute is categorical if its domain size is less than this threshold.
# Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
threshold_value = 10

# specify categorical attributes
# categorical_attributes = {'education': True}
categorical_attributes = {'label': True}

# specify which attributes are candidate keys of input dataset.
# candidate_keys = {'ssn': True}
candidate_keys = {'index': True}

# A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
# change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
# Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
epsilon = 50

# The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
# TODO: change this to 10
degree_of_bayesian_network = 5

# Number of tuples generated in synthetic dataset.
# TODO: change this to 100_000
num_tuples_to_generate = 1_000_000 

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys)
describer.save_dataset_description_to_file(description_file)
display_bayesian_network(describer.bayesian_network)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)