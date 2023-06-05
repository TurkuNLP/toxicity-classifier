## Example usage of scripts

Predictions for new texts can be made with different scripts. `Toxicity_predictions.py` is meant for finding the best threshold for new texts and saves only the highest toxicity prediction for any toxicity label and sorts all the texts. `Simple_multi-predictions.py` is meant for getting the predicted texts with id and all of the probabilities for the labels to .tsv files.

Results for the annotated test set can be gotten through the script `get_test-predictions.py` and it can be used for jsonl and the annotated test set to get thresholded labels with id (and text) for error analysis.

A demo for testing predictions easily is in the file `Toxicity_Demo.ipynb`.



Example usage:

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

