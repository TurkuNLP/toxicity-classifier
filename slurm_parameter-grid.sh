#!/bin/bash

# from Devlin et al. 2018 (https://arxiv.org/pdf/1810.04805.pdf), Sec. A.3
# """
# [...] we found the following range of possible values to work well across all tasks:
# * Batch size: 16, 32
# * Learning rate (Adam): 5e-5, 3e-5, 2e-5
# * Number of epochs: 2, 3, 4
# """

# implemented from script from https://github.com/spyysalo/transformer-ner/blob/main/scripts/slurm-run-parameter-grid.sh

DIR="$( pwd )"
JOBDIR="$DIR/../jobs"

MAX_JOBS=100

mkdir -p "$JOBDIR"

MODELS="
TurkuNLP/bert-base-finnish-cased-v1
"

DATA_DIRS="
data
"

SEQ_LENS="128"

#BATCH_SIZES="2"
BATCH_SIZES="2 4 8 16"
# w/SEQ_LEN 128: BATCH_SIZES="4 8 16 24"

#LEARNING_RATES="2e-5"
LEARNING_RATES="5e-5 3e-5 2e-5"

EPOCHS="2 3"
#EPOCHS="2 3 4"

#REPETITIONS=5
REPETITIONS=3

# for repetition in `seq $REPETITIONS`; do
#     for seq_len in $SEQ_LENS; do
	for batch_size in $BATCH_SIZES; do
	    for learning_rate in $LEARNING_RATES; do
		for epochs in $EPOCHS; do
		    for model in $MODELS; do
			for data_dir in $DATA_DIRS; do
			    while true; do
				jobs=$(ls "$JOBDIR" | wc -l)
				if [ $jobs -lt $MAX_JOBS ]; then break; fi
				echo "Too many jobs ($jobs), sleeping ..."
				sleep 60
			    done
			    echo "Submitting job with params $model $data_dir $batch_size $learning_rate $epochs" #$seq_len
			    job_id=$(
				sbatch "$DIR/slurm-run-dev.sh" \
				    $model \
				    $data_dir \
				    $seq_len \
				    $batch_size \
				    $learning_rate \
				    $epochs \
				    | perl -pe 's/Submitted batch job //'
			    )
			    echo "Submitted batch job $job_id"
			    touch "$JOBDIR"/$job_id
			    sleep 5
			done
		    done
		done
	    done
	done
#     done
# done