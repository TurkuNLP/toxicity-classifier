# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no labels), binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is from the [TurkuNLP github](https://github.com/TurkuNLP/wikipedia-toxicity-data-fi).

Current results for runs can be found [here](https://docs.google.com/spreadsheets/d/1g8Ya5lx80CdqEXHiwKO32EFv2AMNkZzPypzlb1PI6xk/edit?usp=sharing)


## Example usage of scripts

```
python3 toxic_classifier.py --train [FILE] --test [FILE] --model [MODEL_NAME]
```

All the scripts use the same required arguments but optional arguments may vary so to learn more about the arguments that can be used with these scripts do:

```
python3 [SCRIPT] -h
```