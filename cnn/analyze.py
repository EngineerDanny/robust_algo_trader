import pandas as pd
from glob import glob

date_time = "2023-12-23_13:34"
date_time = "2023-12-28_04:39"
date_time = "2024-01-01_19:13"
date_time = "2024-01-01_19:36"
date_time = "2024-01-02_08:30"
date_time = "2024-01-14_18:37"

date_time = "2024-01-14_18:19"
date_time = "2024-01-14_19:06"
date_time = "2024-01-14_19:11"
date_time = "2024-01-14_19:24"
date_time = "2024-01-19_19:14"
date_time = "2024-01-24_00:12"
date_time = "2024-01-24_06:09"

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/cnn/results"
error_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)