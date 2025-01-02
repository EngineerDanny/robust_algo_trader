import pandas as pd
from glob import glob

time = "2024-09-27_03:27" # EUR_USD_M15 - AUD_CAD_M15
time = "2024-09-27_03:35" # EUR_GBP_M15 - EUR_CAD_M15
time = "2024-12-12_12:01"
time = "2024-12-12_14:03"
time = "2024-12-12_15:48"
time = "2024-12-14_04:20"

time = "2024-12-26_13:35"
time = "2024-12-27_17:41"
time = "2024-12-31_13:33"

time = "2025-01-02_08:33"


out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)