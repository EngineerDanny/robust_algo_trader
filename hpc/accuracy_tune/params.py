
from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil
import sys
import pandas as pd


root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data"
params_df_list = []

algo_list = ["LinearRegression"]
# train_size_list = [1680, 3360, 6720, 13_440, 16_800]
# window_size_list = [48, 96, 192, 300, 384]

train_size_list = [33600]
window_size_list = [480]
forecast_horizon_list = [360]
sma_list = [200]
dataset_list = ["EURUSD_H1"]
slope_threshold_list = [0]
year_list = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
# step_length_list = [24]
step_length_list = [1]

for dataset_name in dataset_list:
    params_dict = {
        'dataset_name': [dataset_name],
        'fh': forecast_horizon_list,
        'window_size': window_size_list,
        'algorithm': algo_list,
        'train_size': train_size_list,
        'sma': sma_list,
        
        'step_length': step_length_list,
        'slope_threshold' : slope_threshold_list,
        'year': year_list,
        
    }

    params_df = pd.MultiIndex.from_product(
        params_dict.values(),
        names=params_dict.keys()
    ).to_frame().reset_index(drop=True)
    params_df_list.append(params_df)

params_concat_df = pd.concat(params_df_list, ignore_index=True)


n_tasks, ncol = params_concat_df.shape
date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
job_name = f"ml_project_2_{date_time}"
job_dir = "/scratch/da2343/" + job_name
results_dir = os.path.join(job_dir, "results")
os.system("mkdir -p " + results_dir)
params_concat_df.to_csv(os.path.join(job_dir, "params.csv"), index=False)

run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=64
#SBATCH --error={job_dir}/slurm-%A_%a.out
#SBATCH --output={job_dir}/slurm-%A_%a.out
#SBATCH --job-name={job_name}
cd {job_dir}
python run_one.py $SLURM_ARRAY_TASK_ID
"""
run_one_sh = os.path.join(job_dir, "run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)

run_orig_py = "demo_run.py"
run_one_py = os.path.join(job_dir, "run_one.py")
shutil.copyfile(run_orig_py, run_one_py)

orig_dir = os.path.dirname(run_orig_py)
os.system("mkdir -p " + os.path.join(orig_dir, "results"))
os.system("mkdir -p " + os.path.join(orig_dir, "orders"))

orig_csv = os.path.join(orig_dir, "params.csv")
params_concat_df.to_csv(orig_csv, index=False)

msg = f"""created params CSV files and job scripts, test with
python {run_orig_py}
SLURM_ARRAY_TASK_ID=0 bash {run_one_sh}"""
print(msg)
