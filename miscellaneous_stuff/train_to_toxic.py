import pandas as pd
import json
import argparse


parser = argparse.ArgumentParser(
        description="A script for classifying toxic data in a binary manner",
        epilog="Made by Anni Eskelinen"
    )
parser.add_argument('--train', required=True)
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


data = args.train

# first I need to read the json lines
with open(data, 'r') as json_file:
    json_list = list(json_file)
lines = [json.loads(jline) for jline in json_list]
# there is now a list of dictionaries

df=pd.DataFrame(lines)
df['labels'] = df[label_names].values.tolist()

# change to binary: if toxic 1 if clean 0
# first get sum of labels
df['labels'] = df.labels.map(sum) #df[label_names].sum(axis=1)

# then change bigger than 0 to 1 and 0 stays 0
df.loc[df["labels"] > 0, "labels"] = 1

# here only take the ones with label 1
dftoxic = df[df['labels'] == 1]
dfclean = df[df['labels'] == 0]

dftoxic = dftoxic[['text']]
dfclean = dfclean[['text']]

def fix_newlines(dataframe):
    dataframe = dataframe.replace(r'\n',' ', regex=True) # unix
    dataframe = dataframe.replace(r'\r\n',' ', regex=True) # windows
    dataframe = dataframe.replace(r'\r',' ', regex=True) # mac

    return dataframe

dftoxic = fix_newlines(dftoxic)
dfclean = fix_newlines(dfclean)

dftoxic.to_csv('en_toxic_train.tsv', sep="\t", header=False, index=False)
dfclean.to_csv('en_clean_train.tsv', sep="\t", header=False, index=False)

#textlist = dftoxic['text'].values.tolist()
# with open('toxic_train.txt', 'w') as f:
#     for line in textlist:
#         f.write(f"{line}\n")