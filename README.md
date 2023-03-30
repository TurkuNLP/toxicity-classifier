# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), multiclass/binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is available in [huggingface](https://huggingface.co/datasets/TurkuNLP/wikipedia-toxicity-data-fi) and the model will be available in Huggingface later.

New annotated data for Finnish based on a sample from Suomi24 is available in the folder annotations in the file all_annotations.tsv. The script for evaluating that data is in the folder predictions.

## Examples of mistakes the model makes

Error analysis is coming later.


## Add explanations of the labels here? would make sense and people could use them maybe

The labels used for the Jigsaw toxicity data are: INSERT LIST

The explanations for the labels are --- and can be found on .