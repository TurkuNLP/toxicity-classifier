import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

#python3 miscallenous_stuff/get_test-predictions.py --model "models/finbert-large-deepl" --data "data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "test.tsv"

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
            description="A script for predicting toxic texts based on a toxicity classifier and finding the best threshold",
            epilog="Made by Anni Eskelinen"
        )
parser.add_argument('--model', required=True,
    help="the model name")
parser.add_argument('--data', required=True,
    help="the file name of the raw text to use for predictions")
parser.add_argument('--tokenizer', required=True,
    help="the tokenizer to use for tokenizing new text")
parser.add_argument('--filename', required=True,
    help="the file name to give file resulting from the predictions")
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint


# read the data in
data = args.data

with open(data, 'r') as json_file:
        json_list = list(json_file)
lines = [json.loads(jline) for jline in json_list]

print("number of lines in the file", len(lines))

# use pandas to look at each column
df=pd.DataFrame(lines)

df['labels'] = df[label_names[:-1]].values.tolist()
labels = df['labels'].values.tolist()

df = df[['text', 'id']]
pprint(df[:5])



# instantiate model, this is pretty simple
model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

trainer = transformers.Trainer(
    model=model
) 

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding='max_length', # this got it to work, data_collator could have helped as well?
        max_length=512,
        truncation=True,
    )

dataset = datasets.Dataset.from_pandas(df)

#map all the examples
dataset = dataset.map(tokenize)

ids = dataset["id"]

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions




#7 labels, change predictions into something usable in a file

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
probs = probs.tolist()

# change to only take 6 labels from the 7
new_probs = []
for i in range(len(probs)):
    new_probs.append(probs[i][:-1])
probs = new_probs

print(probs[:10])

threshold = 0.6499999999999999

all_labels = []
# set list to have label if the prob is bigger than the threshold
temp_list = []
for i in range(len(probs)):
    for j in range(len(probs[i])):
        if probs[i][j] > threshold:
            temp_list.append(label_names[j])
    all_labels.append(temp_list)
    temp_list = [] # empty the list before next round

# get also the gold labels from the test set to list
gold_labels = []
temp_list2 = []
for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] == 1:
            temp_list2.append(label_names[j])
    gold_labels.append(temp_list2)
    temp_list2 = [] # empty the list before next round

all = tuple(zip(ids, gold_labels, all_labels))
#pprint(all[:10])

allpredict = [item for item in all]

# get to dataframe
def id_and_label(data):
    df = pd.DataFrame(data, columns=['id', 'gold_labels', 'predicted_labels'])
    return df

all_dataframe = id_and_label(allpredict)

# put to csv so we don't need any new lines taken out
filename = args.filename
all_dataframe.to_csv(filename, sep="\t", index=False) # added sep to make tsv
