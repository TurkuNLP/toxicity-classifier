import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

""" Script for predicting whether a text is toxic or clean. Only prints the text and label. """


# this should prevent any caching problems I might have because caching does not happen anymore
datasets.disable_caching()

label_names = [
    'label_identity_attack',
    'label_insult',
    'label_obscene',
    'label_severe_toxicity',
    'label_threat',
    'label_toxicity',
    'label_clean'
]

# parse arguments
parser = argparse.ArgumentParser(
            description="A script for predicting toxic texts based on a toxicity classifier",
            epilog="Made by Anni Eskelinen"
        )
parser.add_argument('--model', required=True,
    help="the model name")
parser.add_argument('--type', required=True,
    help="which type the model is, multilabel, binary or true-binary")
parser.add_argument('--threshold', type=float, default=0.5,
    help="the threshold for the predictions")
parser.add_argument('--data', required=True,
    help="the file name of the raw text to use for predictions")
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint

# instantiate model, this is pretty simple
model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

trainer = transformers.Trainer(
    model=model
) 

# read the data in
data = args.data

with open(data, 'r') as json_file:
        json_list = list(json_file)
lines = [json.loads(jline) for jline in json_list]

lines = lines[:200000] # CHANGE HOW MANY TEXTS WANT TO USE (could be set as an argument)

# use pandas to look at each column
df=pd.DataFrame(lines)
df = df[['body']]
df.rename(columns = {'body':'text'}, inplace = True) # have to change the column name so this works
pprint(df[:5])

# keep every row except ones with deleted text
#df = df[df["body"].str.contains("[deleted]") == False]

#random.shuffle(texts) # not necessary
print(len(lines))

tokenizer = transformers.AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

def tokenize(example):
    return tokenizer(
        example["text"],
        padding='max_length', # this got it to work, data_collator could have helped as well?
        max_length=512,
        truncation=True,
    )

dataset = datasets.Dataset.from_pandas(df)

texts = dataset["text"]

#map all the examples
dataset = dataset.map(tokenize)

threshold = args.threshold

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions


if args.type == "binary":
    import torch.nn.functional as F
    tensor = torch.from_numpy(predictions)
    probabilities = F.softmax(tensor, dim=1) # turn to probabilities using softmax
    probabilities = probabilities.tolist()
    print(probabilities[:10]) # this is now a tensor with two probabilities per example (two labels)
    print(predictions[:10])


    # THIS
    #preds = predictions.argmax(-1) # the -1 gives the indexes of the predictions, takes the one with the biggest number
     # argmax can be used on the probabilities as well although the tensor needs to changed to numpy array first

    # OR THIS
    # idea that if there is no high prediction for e.g. clean label then we set it to toxic (or the other way around)
    # set p[0] or p[1] depending on which we wanna concentrate on
    preds = [0 if p[1] < threshold else np.argmax(p) for p in probabilities]  # if toxic below 0.5 count as clean (set index to 0)


    # get predicted labels
    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    # lastly use zip to get tuple
    prediction_tuple = tuple(zip(texts, labels))

    # make into list of tuples
    toxic = [item for item in prediction_tuple
          if item[1] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1] == "clean"]

    pprint(prediction_tuple[:100])


elif args.type == "multi":
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    preds = np.zeros(probs.shape)

    # next, use threshold to turn them into integer predictions
    preds[np.where(probs >= threshold)] = 1

    probs = probs.tolist()

    # take the clean label away
    # TODO add something to distinct this from 6 labels vs. 7 which I am currently using
    new_pred = []
    for i in range(len(preds)):
        new_pred.append(preds[i][:-1])
    preds = new_pred

    new_probs = []
    for i in range(len(probs)):
        new_probs.append(probs[i][:-1])
    probs = new_probs

    # the predicted indexes
    pred_label_idxs = [] 
    for vals in preds:
        pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

    # the predicted labels
    labels = [] 
    idx2label = dict(zip(range(6), label_names[:-1]))   # could add clean
    for vals in pred_label_idxs:
        if vals:
            labels.append([idx2label[val] for val in vals])
        else:
            labels.append(vals)

    # set label whether the text is toxic or clean
    pred_label = []
    for i in range(len(preds)):
        if sum(preds[i]) > 0:
            pred_label.append("toxic")
        else:
            pred_label.append("clean")

    all = tuple(zip(texts, pred_label)) 

    pprint(all[:100])


elif args.type == "true-binary":
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(predictions))
    print(predictions[:10])
    print(probabilities[:10])

    # get predictions with the threshold
    y_pred = [1 if prob >= threshold else 0 for prob in probabilities] 
    preds = y_pred

    probabilities = probabilities.tolist() 

    # get predicted labels
    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    prediction_tuple = tuple(zip(texts, labels))

    pprint(prediction_tuple)