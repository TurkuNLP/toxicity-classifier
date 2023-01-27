#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module purge
module load pytorch

srun python3 miscallenous_stuff/get_test-predictions.py --model "models/finbert-large-deepl" --data "data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "test.tsv"