#!/bin/bash
#SBATCH --job-name=translation
#SBATCH --account=project_2000539
#SBATCH --partition=gpu # test
#SBATCH --time=4:00:00 # 5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 5?
#SBATCH --mem-per-cpu=8000 #8000 
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

echo "START: $(date)"

#srun python3 miscallenous_stuff/sentence-splitter.py ../data/train_en.jsonl #../data/test_en.jsonl
srun python3 miscallenous_stuff/translator.py ../data/test-sentence-split.csv

echo "END: $(date)"

###SBATCH --gres=gpu:v100:1 # take out for test partition, set time to 10 minutes also then