## Example usage of scripts

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