#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --time=10:00:00 # 5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=grid_logs/%j.out
#SBATCH --error=../logs/%j.err


export COMET_API_KEY="ARr02oZjXsfNYlAeIFROYSj7O"

remove output and job marker on exit
function on_exit {
    rm -f out-$SLURM_JOBID.tsv
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

# check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 model data_dir batch_size learning_rate epochs"
    exit 1
fi

model="$1"
data_dir="$2"
batch_size="$3"
learning_rate="$4"
epochs="$5"

# symlink logs for this run as "latest"
rm -f logs/latest.out logs/latest.err
ln -s "$SLURM_JOBID.out" "grid_logs/latest.out"
ln -s "$SLURM_JOBID.err" "grid_logs/latest.err"

module purge
module load pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python toxic_classifier.py \
    --model "$model" \
    --learning $learning_rate \
    --epochs $epochs \
    --batch $batch_size \
    --train "$data_dir/train_en.jsonl" \
    --test "$data_dir/test_fi_deepl.jsonl" \
    --clean_as_label \
    --loss \
    --dev \
    #--threshold $TR


# echo -n 'PARAMETERS'$'\t'
# echo -n 'model'$'\t'"$model"$'\t'
# echo -n 'data_dir'$'\t'"$data_dir"$'\t'
# echo -n 'train_batch_size'$'\t'"$batch_size"$'\t'
# echo -n 'learning_rate'$'\t'"$learning_rate"$'\t'
# echo -n 'num_train_epochs'$'\t'"$epochs"$'\t'

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"





# module purge
# module load pytorch 

# export COMET_API_KEY="ARr02oZjXsfNYlAeIFROYSj7O"

# EPOCHS=2 # 4 
# LR=2e-5    # "1e-5 4e-6 5e-6 7e-5 8e-6"
# BATCH=12 #8, 16, 32
# TR=0.6
# MODEL='bert-large-cased' #"TurkuNLP/bert-base-finnish-cased-v1" #"TurkuNLP/bert-large-finnish-cased-v1" #'bert-base-cased' # 'bert-large-cased' # "xlm-roberta-large" #'xlm-roberta-base'
# echo "epochs: $EPOCHS, learning rate: $LR, batch size: $BATCH, prediction treshold: $TR, model: $MODEL "

# #original english
# echo "original english"
# srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_en.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --loss --dev --clean_as_label #--threshold $TR

# # translated deepl
# echo "Translated train and test deepl"
# srun python3 toxic_classifier.py --train data/train_fi_deepl.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --clean_as_label --loss --dev #--threshold $TR

# translated opus-mt
# echo "opus-mt translated train and test"
# srun python3 toxic_classifier.py --train data/train-opus-mt-translated.csv --test data/test-opus-mt-translated3.csv --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --clean_as_label --loss --dev #--threshold $TR


# backtranslation
# echo "backtranslation using original english test set"
# srun python3 toxic_classifier.py --train data/train_en_backtr_deepl.jsonl --test data/test_en.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --clean_as_label --loss --dev #--threshold $TR


# transfer
# echo "transfer with xlmr"
# srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --clean_as_label --loss --dev #--threshold $TR




# BINARY EVALUATION
# echo "binary evaluation"

#original
# echo "original train and test data"
# srun python3 toxic_classifier.py --train data/train_en.jsonl --test data/test_en.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --binary --loss --clean_as_label #--dev --threshold $TR

# multilingual
# echo "multilingual with english and finnish train files"
# srun python3 toxic_classifier.py --train data/train_fi_deepl.jsonl data/train_en.jsonl --test data/test_fi_deepl.jsonl --model xlm-roberta-base --batch $BATCH --epochs $EPOCHS --learning $LR --loss --binary --clean_as_label #--threshold $TR #--dev


#echo "END: $(date)"