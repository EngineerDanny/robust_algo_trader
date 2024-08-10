import pandas as pd
from glob import glob

time = "2024-06-10_12:52" # EUR_USD_M15
time = "2024-06-10_19:47" # GBP_USD_M15
time = "2024-06-10_22:59" # USD_JPY_M15
time = "2024-06-11_04:01" # USD_CHF_M15
time = "2024-06-11_04:14" # USD_CAD_M15
time = "2024-06-11_11:42" # AUD_USD_M15
time = "2024-06-11_23:43" # AUD_JPY_M15
time = "2024-06-11_23:48" # AUD_CAD_M15
time = "2024-06-12_08:42" # EUR_GBP_M15
time = "2024-06-12_08:44" # EUR_JPY_M15
time = "2024-06-12_08:48" # GBP_CHF_M15
time = "2024-06-12_08:52" # GBP_JPY_M15

time = "2024-08-08_02:39" # GBP_USD_M15
time = "2024-08-08_02:38" # EUR_USD_M15
# NEW
time = "2024-08-10_01:16" # USD_JPY_M15
time = "2024-08-10_01:23" # USD_CHF_M15
time = "2024-08-10_01:27" # USD_CAD_M15
time = "2024-08-10_01:28" # AUD_USD_M15
time = "2024-08-10_01:29" # AUD_JPY_M15
time = "2024-08-10_01:32" # AUD_CAD_M15
time = "2024-08-10_01:33" # EUR_GBP_M15
time = "2024-08-10_01:35" # EUR_JPY_M15
time = "2024-08-10_01:36" # GBP_CHF_M15
time = "2024-08-10_01:38" # GBP_JPY_M15

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)