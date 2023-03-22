#!/bin/bash
#SBATCH --job-name=translation
#SBATCH --account=project_2000539
#SBATCH --partition=gputest # test
#SBATCH --time=00:10:00 # 4 or 5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 5?
#SBATCH --mem-per-cpu=8000 #8000 
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

echo "START: $(date)"

srun python3 miscallenous_stuff/translator.py data/split_output_en.jsonl #data/test-sentence-split.csv

echo "END: $(date)"