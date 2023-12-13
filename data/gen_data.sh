#!/bin/bash
#SBATCH --job-name=gendata         
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/projects/genomic-ml/da2343/ml_project_2/data/gen_data.out	
#SBATCH --error=/projects/genomic-ml/da2343/ml_project_2/data/gen_data.err

python /projects/genomic-ml/da2343/ml_project_2/data/gen_trade_data.py
