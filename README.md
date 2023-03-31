# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), multiclass/binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is available in [huggingface](https://huggingface.co/datasets/TurkuNLP/wikipedia-toxicity-data-fi) and a model trained on this data is available [here](https://huggingface.co/TurkuNLP/bert-large-finnish-cased-toxicity). 

New annotated data for Finnish based on a sample from Suomi24 is available in the folder annotations in the file all_annotations.tsv and in [huggingface](https://huggingface.co/datasets/TurkuNLP/Suomi24-toxicity-annotated). The script for evaluating that data is in the folder predictions.

## Examples of mistakes the model makes

Error analysis is coming later.