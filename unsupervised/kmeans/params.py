import os
import shutil
from datetime import datetime
import pandas as pd


def create_batches(total_tasks, batch_size):
    return [
        (i, min(i + batch_size, total_tasks)) for i in range(0, total_tasks, batch_size)
    ]


def create_job_script(job_dir, job_name, start, end, batch_id):
    return f"""#!/bin/bash
#SBATCH --array={start}-{end-1}
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --error={job_dir}/slurm-%A_%a.out
#SBATCH --output={job_dir}/slurm-%A_%a.out
#SBATCH --job-name={job_name}_batch_{batch_id}
cd {job_dir}
python run_one.py $SLURM_ARRAY_TASK_ID
"""


def main():
    # Setup hyperparameters
    params_dict = {
        'max_cluster_labels': [1],
        'price_history_length': [24],
        'num_perceptually_important_points': [5, 6],
        'distance_measure': [1],
        'num_clusters': [70, 80, 90, 100, 110, 120],
        'atr_multiplier': [10],
        'clustering_algorithm': ['kmeans', 'gaussian_mixture'],
        # 'random_seed': np.arange(1, 100),
        'random_seed': [1, 2, 4, 7, 10, 12, 15, 18, 20, 21, 42, 50, 80, 90, 100, 200, 300],
        'train_period': [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20], # weeks   
        'test_period': [2] # weeks
    }
    params_df = (
        pd.MultiIndex.from_product(params_dict.values(), names=params_dict.keys())
        .to_frame()
        .reset_index(drop=True)
    )

    # Setup job directory
    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    job_name = f"ml_project_2_{date_time}"
    job_dir = os.path.join("/scratch/da2343", job_name)
    results_dir = os.path.join(job_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save parameters
    params_df.to_csv(os.path.join(job_dir, "params.csv"), index=False)

    # Setup batches
    n_tasks = len(params_df)
    batch_size = 1000  # Adjust this value based on your system's capacity
    batches = create_batches(n_tasks, batch_size)

    # Create submission script
    submit_all_contents = "#!/bin/bash\n\n"

    for batch_id, (start, end) in enumerate(batches):
        job_script = create_job_script(job_dir, job_name, start, end, batch_id)
        submit_all_contents += f"""
jobid_{batch_id}=$(sbatch --parsable <<EOF
{job_script}
EOF
)

echo "Submitted batch {batch_id} (jobs {start}-{end-1}) with job ID $jobid_{batch_id}"

"""
        if batch_id < len(batches) - 1:
            submit_all_contents += f"next_jobid_{batch_id+1}=$(sbatch --parsable --dependency=afterok:$jobid_{batch_id})\n\n"

    # Save and make executable the submission script
    submit_all_sh = os.path.join(job_dir, "submit_all_batches.sh")
    with open(submit_all_sh, "w") as submit_all_f:
        submit_all_f.write(submit_all_contents)
    os.chmod(submit_all_sh, 0o755)

    # Copy necessary files
    run_orig_py = "demo_run_optimized.py"
    shutil.copy(run_orig_py, os.path.join(job_dir, "run_one.py"))

    # Print instructions
    print(
        f"""Created params CSV files and a single submission script for {len(batches)} batches.
To submit all jobs, run:
bash {submit_all_sh}
This will submit all batches, with each batch starting after the previous one completes.
The last batch may be smaller if the total number of tasks ({n_tasks}) is not divisible by the batch size ({batch_size}).
To test a single task, you can run:
python {run_orig_py}
SLURM_ARRAY_TASK_ID=<task_id> bash {submit_all_sh}
Replace <task_id> with a number between 0 and {n_tasks-1}."""
    )


if __name__ == "__main__":
    main()
