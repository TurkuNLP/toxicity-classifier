## Example usage of scripts

To train the models use the scripts as follows without optional arguments.

```
python3 toxic_classifier.py --train [FILE] --test [FILE] --model [MODEL_NAME]
```

All the scripts use the same required arguments but optional arguments may vary so to learn more about the arguments that can be used with these scripts do:

```
python3 [SCRIPT] -h
```

Example of arguments for toxic_classifier.py:

```
usage: toxic_classifier.py [-h] --train TRAIN --test TEST --model MODEL
                           [--batch BATCH] [--epochs EPOCHS]
                           [--learning LEARNING] [--threshold THRESHOLD]
                           [--loss] [--dev] [--clean_as_label] [--binary]

A script for classifying toxic data (multi-label, includes binary evaluation
on top)

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN
  --test TEST
  --model MODEL
  --batch BATCH         The batch size for the model
  --epochs EPOCHS       The number of epochs to train for
  --learning LEARNING   The learning rate for the model
  --threshold THRESHOLD
                        The treshold which to use for predictions, used in
                        evaluation. If no threshold is given threshold
                        optimization is used.
  --loss                If used different class weights are used for the loss
                        function
  --dev                 If used the train set is split into train and dev sets
  --clean_as_label      If used the clean examples (no label) are marked as
                        having a label instead for class weights purposes
  --binary              If used the evaluation uses a binary classification
                        (toxic or not) based on the multi-label predictions.

Made by Anni Eskelinen
```