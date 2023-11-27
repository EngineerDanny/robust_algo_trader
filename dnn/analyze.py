# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob
import plotnine as p9

date_time = "2023-11-18_00:22"
date_time = "2023-11-18_00:44"
date_time = "2023-11-23_14:07"
date_time = "2023-11-23_14:14"

out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_3_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
loss_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_3/code/optimization/results"
loss_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)


# create the plot
loss_df.index = range(len(loss_df))
set_colors = {"subtrain": "red", "validation": "blue"}
validation_df = loss_df.query("set_name=='validation'")
min_i = validation_df.loss.argmin()
min_row = pd.DataFrame(dict(validation_df.iloc[min_i, :]), index=[0])
gg = p9.ggplot() +\
    p9.facet_grid(". ~ hidden_layers+step_size", labeller='label_both') +\
    p9.scale_color_manual(values=set_colors) +\
    p9.scale_fill_manual(values=set_colors) +\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df) +\
    p9.geom_point(
        p9.aes(
            x="epoch",
            y="loss",
            fill="set_name"
        ),
        color="black",
        data=min_row)
gg.save(f"loss_df_{date_time}.png", width=20, height=7, dpi=500)