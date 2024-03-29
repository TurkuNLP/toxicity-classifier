import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

""" This script is meant for looking at multi-label predictions for raw text data and saving highest probability with text/id.
"""

# this should prevent any caching problems I might have because caching does not happen anymore
datasets.disable_caching()


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
parser.add_argument('--lines', type=int,
    help="how many lines to predict on, starting from the beginning of file")
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint


# read the data in
data = args.data

if ".json" in data:
    with open(data, 'r') as json_file:
            json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]

    # use pandas to look at each column
    df=pd.DataFrame(lines)

elif ".tsv" in data:
    with open(data, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].split("\t")
        assert len(lines[i]) == 3

    df=pd.DataFrame(lines, columns = ['id', 'label', 'text'])

# line_amount = args.lines
# print("number of lines in the file", len(lines))
# lines = lines[:line_amount] 



# If I want the reddit data to use this I need to do these changes
if "reddit" in data:
    df.rename(columns = {'body':'text'}, inplace = True) # have to change the column name so this works
    # keep every row except ones with deleted text
    df = df[df.text != "[deleted]"] 

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

texts = dataset["text"]
ids = dataset["id"]

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions



sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
probs = probs.tolist()


# get probabilities to their own lists for use in tuple and then dataframe (each with their own column)
identity_attack = [probs[i][0] for i in range(len(probs))]
insult = [probs[i][1] for i in range(len(probs))]
obscene = [probs[i][2] for i in range(len(probs))]
severe_toxicity = [probs[i][3] for i in range(len(probs))]
threat = [probs[i][4] for i in range(len(probs))]
toxicity  = [probs[i][5] for i in range(len(probs))]


all = tuple(zip(ids, identity_attack, insult, obscene, severe_toxicity, threat, toxicity)) # texts, highprob
allwithtexts = tuple(zip(ids, texts, identity_attack, insult, obscene, severe_toxicity, threat, toxicity)) # texts, highprob
#pprint(all[:10])

allpredict = [item for item in all]
alltextspredict = [item for item in allwithtexts]

# get to dataframe
def id_and_label(data):
    df = pd.DataFrame(data, columns=['id', 'identity_attack', 'insult', 'obscene', 'severe_toxicity', 'threat', 'toxicity']) # 'text', 'probability'
    return df

def text_and_label(data):
    df = pd.DataFrame(data, columns=['id', 'text', 'identity_attack', 'insult', 'obscene', 'severe_toxicity', 'threat', 'toxicity']) # 'probability'
    return df


# id and labels + their probabilities
all_dataframe = id_and_label(allpredict)

# put to csv so we don't need any new lines taken out
filename = args.filename
all_dataframe.to_csv(filename, sep="\t", index=False) # added sep to make tsv


# # text, labels and their probabilities

# alltexts_dataframe = text_and_label(alltextspredict)

# # change csv to not have new lines
# alltexts_dataframe["text"] = alltexts_dataframe["text"].str.replace("\n", "<NEWLINE>")
# alltexts_dataframe["text"] = alltexts_dataframe["text"].str.replace("\r\n", "<NEWLINE>")
# alltexts_dataframe["text"] = alltexts_dataframe["text"].str.replace("\r", "<NEWLINE>")
# alltexts_dataframe["text"] = alltexts_dataframe["text"].str.replace("\t", "<TAB>")

# alltexts_dataframe.to_csv("predictions/reddit-maybe-works.csv", index=False)


