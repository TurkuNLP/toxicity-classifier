#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

EPOCHS=3 
LR=8e-6    # "1e-5 4e-6 5e-6 7e-5 8e-6"
BATCH=8
TR=0.5

echo "epochs: $EPOCHS, learning rate: $LR, batch size: $BATCH, prediction treshold: $TR"

#TRANSLATED
# echo "Translated train and test"
# srun python3 toxic_classifier.py --train data/train_fi_deepl.jsonl --test data/test_fi_deepl.jsonl --model TurkuNLP/bert-base-finnish-cased-v1 --batch $BATCH --epochs $EPOCHS --learning $LR --treshold $TR

# should I use xlmr?


#ORIGINAL
echo "original train and test data"
srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_en.jsonl --model bert-base-cased --batch $BATCH --epochs $EPOCHS --learning $LR --treshold $TR