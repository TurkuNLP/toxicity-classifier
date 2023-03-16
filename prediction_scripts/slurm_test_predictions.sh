#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gputest
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=../logs/%j.out
#SBATCH --error=../../logs/%j.err

echo "START $(date)"

module purge
module load pytorch
# 0.6499999999999999 
# "../models/finbert-large-deepl"

srun python3 get_test-predictions.py --new_test --threshold 0.5 --model "../models/finbert-multi/good_finbert_based_model" --data "../data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "testgoodfinbertfix.tsv"

#srun python3 get_test-predictions.py --new_test --threshold 0.5 --model "../models/transfer-devset" --data "../data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "testxlmr.tsv"

#srun python3 get_test-predictions.py --new_test --threshold 0.6499999999999999 --model "../models/opus-mt-finbert-base" --data "../data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "test-opus.tsv"

echo "END: $(date)