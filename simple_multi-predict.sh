#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=1:30:00 # depends highly on how many examples I want to see predicted (1h for base, 2h 30min for large, 10h for all on base)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 # from 10 to 5?
#SBATCH --mem-per-cpu=30G #8000 
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --output=logs/predictions/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

echo "START: $(date)"

LINES=2000

#python3 simple_multi-predictions.py --model models/multi-toxic-tr-optimized --data data/ylenews-fi-2011-2018.dedup.filtered.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES

# this file is massive at 68G, whereas reddit was 3,2G
python3 simple_multi-predictions.py --model models/multi-toxic-tr-optimized --data data/suomi24-2001-2020.dedup.filtered.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES

echo "END: $(date)"