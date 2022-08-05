# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is from the [TurkuNLP github](https://github.com/TurkuNLP/wikipedia-toxicity-data-fi).

Current results for runs can be found [here](https://docs.google.com/spreadsheets/d/1g8Ya5lx80CdqEXHiwKO32EFv2AMNkZzPypzlb1PI6xk/edit?usp=sharing) and the overview of everything is [here](https://docs.google.com/document/d/1ht2dqMYe8p5lDqYE2kWrz-F-RUeOfh60J8J925SeP8g/edit?usp=sharing)


## Example usage of scripts

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

possible datasets to use in the future [hate speech dataset](https://github.com/Vicomtech/hate-speech-dataset), [ALONE](https://arxiv.org/abs/2008.06465), [ToxiGen](https://huggingface.co/datasets/skg/toxigen-data), [Surge ai Categorized Toxicity Dataset](https://app.surgehq.ai/datasets/categorized-toxicity?token=faVpwAMXx_VAZI4p)
