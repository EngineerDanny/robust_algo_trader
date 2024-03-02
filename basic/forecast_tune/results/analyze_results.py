import pandas as pd
import numpy as np
import plotnine as p9
import matplotlib.pyplot as plt

date_time = "2023-07-05_19:33"

root_results_dir = "/projects/genomic-ml/da2343/ml_project_2/hpc/results"

error_df = pd.read_csv(f"{root_results_dir}/{date_time}_results.csv")

df = error_df.copy() 

# Set the index column as the index of the dataframe
df.set_index("index", inplace=True)

# Plot a bar chart of the MAPE column
df["MAPE"].plot(kind="bar", color="blue", title="MAPE by index")
plt.xlabel("index")
plt.ylabel("MAPE")
plt.show()

# save the plot to a file
plt.savefig(f"{root_results_dir}/{date_time}_results.png", dpi=300)
