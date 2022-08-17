#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=01:00:00 # depends highly on how many examples I want to see predicted
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 # from 10 to 5?
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/predictions/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

echo "START: $(date)"

#type multi, binary, true-binary

#srun python3 toxicity_predictions.py --model models/multi-toxic --type multi --threshold 0.765 --data data/reddit-Suomi.jsonl

#srun python3 toxicity_predictions.py --model models/binary-toxic --type binary --threshold 0.75 --data data/reddit-Suomi.jsonl

#srun python3 toxicity_predictions.py --model models/true-binary-toxic --type true-binary --threshold 0.75 --data data/reddit-Suomi.jsonl

python3 simple_predict.py --model models/multi-toxic-tr-optimized --type multi --threshold 0.765 --data data/reddit-Suomi.jsonl

echo "END: $(date)"