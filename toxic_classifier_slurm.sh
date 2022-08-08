#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=05:00:00 # 5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

EPOCHS=4 
LR=8e-6    # "1e-5 4e-6 5e-6 7e-5 8e-6"
BATCH=8
TR=0.6
MODEL="TurkuNLP/bert-base-finnish-cased-v1" #"TurkuNLP/bert-base-finnish-cased-v1" #'bert-base-cased' # "xlm-roberta-large" #'xlm-roberta-base'
echo "epochs: $EPOCHS, learning rate: $LR, batch size: $BATCH, prediction treshold: $TR, model: $MODEL "

#translated
# echo "Translated train and test"
# srun python3 toxic_classifier.py --train data/train_fi_deepl.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --loss #--threshold $TR  #--dev

#original
# echo "original train and test data"
# srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_en.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --threshold $TR --loss #--dev

# transfer
# echo "transfer from english train to translated finnish test"
# srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --threshold $TR --loss --clean_as_label #--dev


# BINARY EVALUATION
# echo "binary evaluation"

#translated
# echo "Translated train and test"
# srun python3 toxic_classifier.py --train data/train_fi_deepl.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --threshold $TR --loss --binary --clean_as_label #--dev

# transfer
echo "transfer from english train to translated finnish test"
srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --threshold $TR --loss --binary --clean_as_label #--dev



echo "END: $(date)"