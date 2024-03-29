import pandas as pd
from glob import glob

time = "2024-03-14_14:51"
time = "2024-03-14_17:26"
time = "2024-03-14_18:18"
time = "2024-03-14_19:00"
time = "2024-03-14_20:08"
time = "2024-03-15_10:31"
time = "2024-03-15_13:27"
time = "2024-03-15_15:44"
time = "2024-03-15_16:27"
time = "2024-03-16_13:35"
time = "2024-03-18_09:34"
time = "2024-03-18_16:10"
time = "2024-03-19_00:56"
time = "2024-03-20_04:35"
time = "2024-03-20_08:28"
time = "2024-03-20_13:00"

time = "2024-03-21_07:59"
time = "2024-03-21_10:50"
time = "2024-03-21_20:28"
time = "2024-03-21_21:59"
time = "2024-03-22_19:23"
time = "2024-03-23_02:14"
time = "2024-03-23_07:49"
time = "2024-03-25_01:47"

time = "2024-03-29_04:53"

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/kmeans/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)