#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=12:00:00 # depends highly on how many examples I want to see predicted (1h for base, 2h 30min for large, 10h for all on base)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 # from 10 to 5?
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/predictions/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

echo "START: $(date)"

LINES=2000

#type multi, multi-base, binary, true-binary

#srun python3 toxicity_predictions.py --model models/multi-toxic --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES

#srun python3 toxicity_predictions.py --model models/multilingual-multi-tr --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer xlm-roberta-base --lines $LINES

#srun python3 toxicity_predictions.py --model models/binary-toxic --type binary --data data/reddit-Suomi.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES #--threshold 0.75

#srun python3 toxicity_predictions.py --model models/true-binary-toxic-tr --type true-binary --threshold 0.75 --data data/reddit-Suomi.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES

#srun python3 toxicity_predictions.py --model models/xlmr-nocleanlabel-tr --type multi-base --threshold 0.75 --data data/reddit-Suomi.jsonl --tokenizer xlm-roberta-base --lines $LINES

python3 toxicity_predictions.py --model models/multi-toxic-tr-optimized --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 #--lines $LINES

#srun python3 toxicity_predictions.py --model models/multi-toxic-large-tr --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer TurkuNLP/bert-base-finnish-cased-v1 --lines $LINES # finbert-large 

#srun python3 toxicity_predictions.py --model models/xlmr-large-multi-tr --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer xlm-roberta-large --lines $LINES # xlmr-large  

# transfer
#srun python3 toxicity_predictions.py --model models/xlmr-transfer --type multi --threshold 0.76 --data data/reddit-Suomi.jsonl --tokenizer xlm-roberta-base --lines $LINES

echo "END: $(date)"