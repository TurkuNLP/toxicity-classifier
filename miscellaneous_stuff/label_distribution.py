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
test_toxic = dftest[dftest[label_names].sum(axis=1) > 0]
test_clean = dftest[dftest[label_names].sum(axis=1) == 0]
print("all lines:", len(train_toxic)+len(train_clean)+ len(test_toxic)+ len(test_clean))
print("all toxic percent:", (len(train_toxic) + len(test_toxic)) / (len(train_clean) + len(test_clean)))
print("toxic train: ", len(train_toxic))
print("clean train: ", len(train_clean))
print("train toxic percent:" , len(train_toxic) / len(train_clean))
print("toxic test: ", len(test_toxic))
print("clean test: ", len(test_clean))
print("test toxic percent:", len(test_toxic) / len(test_clean))

plot2 = pd.DataFrame(dict(
  train_toxic=[len(train_toxic)], 
  train_clean=[len(train_clean)],
  test_toxic=[len(test_toxic)], 
  test_clean=[len(test_clean)]
)).plot(kind='barh')
fig = plot2.get_figure()
fig.savefig('clean_toxic.jpg')
