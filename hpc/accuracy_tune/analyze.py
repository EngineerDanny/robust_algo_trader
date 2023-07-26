# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob


date_time = "2023-07-05_20:29"
date_time = "2023-07-10_19:18"
date_time = "2023-07-10_19:51"
date_time = "2023-07-10_21:09"
date_time = "2023-07-24_14:48"

date_time = "2023-07-24_17:04"
date_time = "2023-07-24_17:52"
date_time = "2023-07-24_19:57"
date_time = "2023-07-24_20:34"
date_time = "2023-07-24_21:00"


out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/hpc/accuracy_tune/results"
error_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)
