import pandas as pd
from glob import glob

time = "2025-01-16_23:42"
time = "2025-01-17_12:11"
time = "2025-01-18_09:53"

time = "2025-01-19_01:28"
time = "2025-01-19_08:02"

time = "2025-01-20_17:15"
time = "2025-01-20_18:26"
time = "2025-01-21_11:28"
time = "2025-01-24_18:13"
time = "2025-01-24_20:45"
time = "2025-01-26_21:55"

out_df_list = []

try:
    for out_csv in glob(f"/scratch/da2343/ml_project_2_{time}/results/*.csv"):
        out_df_list.append(pd.read_csv(out_csv))
    error_df = pd.concat(out_df_list)

    root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/unsupervised/kmeans/results"
    error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)
except Exception as e:
    print(e)
    print(out_csv)
    