import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

""" Script for predicting whether a text is toxic or clean. Only prints the text and label and saves toxic, clean or all to either .csv or .tsv files. """


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
parser.add_argument('--tokenizer', required=True,
    help="the tokenizer to use for tokenizing new text")
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

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

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
    all = tuple(zip(texts, labels))

    pprint(all[:100])


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

    # get predictions with the threshold
    y_pred = [1 if prob >= threshold else 0 for prob in probabilities] 
    preds = y_pred

    # get predicted labels
    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    all = tuple(zip(texts, labels))

    pprint(all[:100])


# make tuple to dataframe and then save to a file

# if want to save purely toxic or clean texts
toxic = [item for item in all
        if item[1] == "toxic"]
clean = [item for item in all
        if item[1] == "clean"]

# only text to file
def only_toxic_clean(data):
    df = pd.DataFrame(list(data), columns=['text', 'label'])
    df = df['text']
    return df

# text and label to file
def text_and_label(data):
    df = pd.DataFrame(list(data), columns=['text', 'label'])
    return df

#dataframe = text_and_label(all) 
dataframe = only_toxic_clean(clean) # clean or toxic, could put both but to different files
dataframe2 = only_toxic_clean(toxic)


# unfortunately all new lines are actual new lines in the csv files not \n so something should maybe be done about it? TODO
# to csv
#dataframe.to_csv('predictions/predicted.csv')

# or to tsv
# dataframe.to_csv('predictions/clean_predicted.tsv', sep="\t", header=False) #, index=False, (this includes the row number now (to make it clear which example is which) and text)
# dataframe.to_csv('predictions/toxic_predicted.tsv', sep="\t", header=False) #, index=False, (this includes the row number now (to make it clear which example is which) and text)


textlist = dataframe['text'].values.tolist()
# or to txt file
with open('predictions/clean_predicted.txt', 'w') as f:
    for line in textlist:
        f.write(f"{line}\n")

textlist2 = dataframe2['text'].values.tolist()
with open('predictions/toxic_predicted.txt', 'w') as f:
    for line in textlist2:
        f.write(f"{line}\n")