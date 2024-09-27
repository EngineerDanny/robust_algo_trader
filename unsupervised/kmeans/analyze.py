import pandas as pd
from glob import glob

time = "2024-08-15_21:03" # EUR_USD_M15
time = "2024-08-15_21:08" # GBP_USD_M15
time = "2024-08-15_21:11" # USD_JPY_M15
time = "2024-08-15_21:13" # USD_CHF_M15
time = "2024-08-15_21:15" # AUD_JPY_M15
time = "2024-08-15_21:18" # USD_CAD_M15
time = "2024-08-15_21:19" # AUD_USD_M15
time = "2024-08-15_21:22" # AUD_CAD_M15
time = "2024-08-15_21:23" # EUR_GBP_M15 
time = "2024-08-15_21:25" # EUR_JPY_M15
time = "2024-08-15_21:27" # GBP_CHF_M15
time = "2024-08-15_21:31" # GBP_JPY_M15
time = "2024-08-15_21:33" # EUR_CHF_M15
time = "2024-08-15_21:34" # AUD_NZD_M15
time = "2024-08-15_21:36" # CAD_JPY_M15
time = "2024-08-15_21:38" # NZD_USD_M15
time = "2024-08-15_21:55" # EUR_CAD_M15

time = "2024-09-27_03:27" # EUR_USD_M15 - AUD_CAD_M15
time = "2024-09-27_03:35" # EUR_GBP_M15 - EUR_CAD_M15

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)