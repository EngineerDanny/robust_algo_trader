from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil
import sys
import pandas as pd
import json

# Load the config file
config_path = "/projects/genomic-ml/da2343/ml_project_2/settings/config.json"
with open(config_path) as f:
  config = json.load(f) 
config_settings = config["trading_settings"]

params_df_list = []
data_dict = {
    'max_cluster_labels': ['Alice', 'Bob', 'Charlie', 'David'],
    'price_history_length': [25, 30, 35, 28],
    'num_perceptually_important_points': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'distance_measure': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'num_clusters': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'atr_multiplier': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'clustering_algorithm': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'random_seed': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'train_period': ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'test_period': 4 * [10],
}
params_concat_df = pd.DataFrame(data_dict)

n_tasks, ncol = params_concat_df.shape
date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
job_name = f"ml_project_2_{date_time}"
job_dir = "/scratch/da2343/" + job_name
results_dir = os.path.join(job_dir, "results")
os.system("mkdir -p " + results_dir)
params_concat_df.to_csv(os.path.join(job_dir, "params.csv"), index=False)

run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=24:00:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --error={job_dir}/slurm-%A_%a.out
#SBATCH --output={job_dir}/slurm-%A_%a.out
#SBATCH --job-name={job_name}
cd {job_dir}
python run_one.py $SLURM_ARRAY_TASK_ID
"""
run_one_sh = os.path.join(job_dir, "run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)

run_orig_py = "demo_monte_carlo.py"
run_one_py = os.path.join(job_dir, "run_one.py")
shutil.copyfile(run_orig_py, run_one_py)
orig_dir = os.path.dirname(run_orig_py)
orig_results = os.path.join(orig_dir, "results")
os.system("mkdir -p " + orig_results)
orig_csv = os.path.join(orig_dir, "params.csv")
params_concat_df.to_csv(orig_csv, index=False)

msg = f"""created params CSV files and job scripts, test with
python {run_orig_py}
SLURM_ARRAY_TASK_ID=0 bash {run_one_sh}"""
print(msg)