# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), multiclass/binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is available in [huggingface](https://huggingface.co/datasets/TurkuNLP/wikipedia-toxicity-data-fi) and the model will be available in Huggingface later.

## Examples of mistakes the model makes

Error analysis is coming later.


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

Predictions for new texts can be made in two different scripts located in the prediction scripts folder, toxicity_predictions.py and simple_multi-predictions.py. Toxicity_predictions.py is meant for finding the best threshold for new texts and saves only the highest toxicity prediction for any toxicity label and sorts all the texts. Simple_multi-predictions.py is meant for getting the predicted texts with id and all of the probabilities for the labels to .tsv files.

Both scripts use the same arguments.

```
usage: simple_multi-predictions.py [-h] --model MODEL --data DATA --tokenizer
                                   TOKENIZER [--lines LINES]

A script for predicting toxic texts based on a toxicity classifier and finding
the best threshold

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the model name
  --data DATA           the file name of the raw text to use for predictions
  --tokenizer TOKENIZER
                        the tokenizer to use for tokenizing new text
  --lines LINES         how many lines to predict on, starting from the
                        beginning of file

Made by Anni Eskelinen
```

Results for the annotated test set can be gotten through the script get_test-predictions.py which is also located in the folder prediction_scripts.