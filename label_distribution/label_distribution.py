import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(
        description="A script for getting the label distributions",
        epilog="Made by Anni Eskelinen"
    )
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)
args = parser.parse_args()
print(args)


label_names = [
    'label_identity_attack',
    'label_insult',
    'label_obscene',
    'label_severe_toxicity',
    'label_threat',
    'label_toxicity'
]

def json_to_df(data):
    # first I need to read the json lines
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries

    df=pd.DataFrame(lines)
    return df


dftrain = json_to_df(args.train)
dftest = json_to_df(args.test)

plot1 = dftrain[label_names].sum().sort_values().plot(kind="barh")
fig = plot1.get_figure()
fig.savefig('toxic_labels.jpg')

train_toxic = dftrain[dftrain[label_names].sum(axis=1) > 0]
train_clean = dftrain[dftrain[label_names].sum(axis=1) == 0]

plot2 = pd.DataFrame(dict(
  toxic=[len(train_toxic)], 
  clean=[len(train_clean)]
)).plot(kind='barh')
fig = plot2.get_figure()
fig.savefig('clean_toxic.jpg')
