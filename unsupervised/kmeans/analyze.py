import pandas as pd
from glob import glob


time = "2024-03-29_04:53"
time = "2024-03-29_07:23"
time = "2024-03-31_15:54"
time = "2024-03-31_16:37"
time = "2024-03-31_17:12"
time = "2024-04-01_15:42"
time = "2024-04-01_17:06"
time = "2024-04-01_18:11"
time = "2024-04-16_16:47"
time = "2024-04-18_19:23"
time = "2024-04-18_20:30"
time = "2024-04-19_01:29"
time = "2024-04-19_09:39"
time = "2024-05-19_16:34"
time = "2024-05-19_16:58"
time = "2024-05-19_17:38"
time = "2024-05-19_18:55"
time = "2024-05-20_04:15"
time = "2024-05-20_10:30"
time = "2024-06-10_12:52" # EUR_USD_M15
time = "2024-06-10_19:47" # GBP_USD_M15
time = "2024-06-10_22:59" # USD_JPY_M15
time = "2024-06-11_04:01" # USD_CHF_M15
# time = "2024-06-11_04:14" # USD_CAD_M15
# time = "2024-06-11_11:42" # AUD_USD_M15

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)