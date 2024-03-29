#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gputest
#SBATCH --time=00:10:00 # depends highly on how many examples I want to see predicted (1h for base, 2h 30min for large, 10h for all on base)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 5? even less lol
#SBATCH --mem-per-cpu=8000 #8000 
#SBATCH --gres=gpu:v100:1
#SBATCH --output=../logs/predictions/%j.out
#SBATCH --error=../../logs/%j.err

module load pytorch 

echo "START: $(date)"

LINES=200000

#srun python3 simple_multi-predictions.py --model models/multi-toxic-tr-optimized --data data/ylenews-fi-2011-2018.dedup.filtered.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --filename predictions/yle.csv--lines $LINES

srun python3 simple_multi-predictions.py --model "../models/xlmr/xlmr-large-multi-tr" --data "../annotations/all_annotations.tsv" --tokenizer "xlm-roberta-base" --filename "../predictions/annotated-transfer.tsv" #--lines $LINES

#srun python3 simple_multi-predictions.py --model models/multi-toxic-tr-optimized --data data/suomi24-2001-2020-1p-sample.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --filename predictions/s24predictions.tsv #--lines $LINES


# this file is massive at 68G, whereas reddit was 3,2G
#srun python3 simple_multi-predictions.py --model models/multi-toxic-tr-optimized --data data/suomi24-2001-2020.dedup.filtered.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES

echo "END: $(date)"