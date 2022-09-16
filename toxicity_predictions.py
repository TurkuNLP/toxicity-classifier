import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

""" This script is meant for looking at predictions for raw text and changing the for what classifies as clean and what toxic manually by looking at the neutral texts.
This script can be used on three different saved models, multi-label with 7 labels, multiclass with two labels and binary. 
To make predictions with the script, specify the data, the model, the type of model and the threshold.
"""

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
parser.add_argument('--type', required=True,
    help="which type the model is, multilabel, binary or true-binary")
parser.add_argument('--threshold', type=float, default=0.5,
    help="the threshold for the predictions")
parser.add_argument('--data', required=True,
    help="the file name of the raw text to use for predictions")
parser.add_argument('--tokenizer', required=True,
    help="the tokenizer to use for tokenizing new text")
parser.add_argument('--lines', type=int,
    help="how many lines to predict on, starting from the beginning of file")
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

line_amount = args.lines
print("number of lines in the reddit file", len(lines))
lines = lines[:line_amount] 

# use pandas to look at each column
df=pd.DataFrame(lines)
df = df[['body']]
df.rename(columns = {'body':'text'}, inplace = True) # have to change the column name so this works
pprint(df[:5])

# keep every row except ones with deleted text
df = df[df["text"] != "[deleted]"] # .str.contains("[deleted]") == False

print("number of lines after deleting [deleted] rows", df.shape[0])


#random.shuffle(texts) # not necessary

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

threshold = args.threshold

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions

# OOOR pipeline with 'return_all_scores' as parameter would do the same thing as above
# https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline 

# what to do with the predictions depending on the type
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
    preds = [0 if p[1] < threshold else np.argmax(p) for p in probabilities]  # if toxic below threshold count as clean (set index to 0)

    # get all labels and their probabilities
    all_label_probs = []
    for prob in probabilities:
        all_label_probs.append(tuple(zip(prob, ["clean", "toxic"])))

    # get predicted labels
    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    # now just loop to a list, get the probability with the indexes from preds
    probs = []
    for i in range(len(probabilities)):
        probs.append(probabilities[i][preds[i]]) # preds[i] gives the correct index for the current probability

    labelprob = tuple(zip(labels, probs))

    # lastly use zip to get tuple
    prediction_tuple = tuple(zip(texts, labelprob, all_label_probs))

    # make into list of tuples
    toxic = [item for item in prediction_tuple
          if item[1][0] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1][0] == "clean"]

    # now sort by probability, descending
    toxic.sort(key = lambda x: float(x[1][1]), reverse=True)
    clean.sort(key = lambda x: float(x[1][1]), reverse=True)
    clean2 = sorted(clean, key = lambda x: float(x[1][1])) # ascending

    # beginning most toxic, middle "neutral", end most clean
    all = toxic + clean2

    print("TOXIC")
    pprint(toxic[:10])
    print("NEUTRAL")
    pprint(toxic[-20:]) # these two middle are the closest to "neutral" where the threshold is
    pprint(clean2[:20]) # clean[-20:]
    print("CLEAN")
    pprint(clean[:10])

    # get the most toxic to tsv file
    toxicity  = [(toxic[i][0], toxic[i][1][0], toxic[i][1][1]) for i in range(len(toxic))]   
    cleaned  = [(clean[i][0], clean[i][1][0], clean[i][1][1]) for i in range(len(clean))]
    allpredict = [(all[i][0], all[i][1][0], all[i][1][1]) for i in range(len(all))]

# 6 or 7 labels
elif args.type == "multi" or args.type == "multi-base":
    # if I want to setup pipeline I have to set function to apply to sigmoid manually (not fun for this, works out of the box for multiclass)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    preds = np.zeros(probs.shape)

    # next, use threshold to turn them into integer predictions
    preds[np.where(probs >= threshold)] = 1

    probs = probs.tolist()

    if args.type == "multi":
        # take the clean label away
        new_pred = []
        for i in range(len(preds)):
            new_pred.append(preds[i][:-1])
        preds = new_pred

        new_probs = []
        for i in range(len(probs)):
            new_probs.append(probs[i][:-1])
        probs = new_probs

    print(probs[:10])
    print(preds[:10])

    # put all the probabilities to a list with all the labels
    prob_label_tuples = []
    for prob in probs:
        prob_label_tuples.append(tuple(zip(prob, label_names[:-1])))

    # the predicted indexes
    pred_label_idxs = [] 
    for vals in preds:
        pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

    # all the probabilities for the predicted labels
    probs_picked = []
    for i in range(len(probs)):
        if pred_label_idxs[i]:
            probs_picked.append([probs[i][val] for val in pred_label_idxs[i]])
        else:
            probs_picked.append(pred_label_idxs[i])

    # the predicted labels
    labels = [] 
    idx2label = dict(zip(range(6), label_names[:-1]))   # could add clean
    for vals in pred_label_idxs:
        if vals:
            labels.append([idx2label[val] for val in vals])
        else:
            labels.append(vals)

    # the predicted labels and their probabilities
    predicted = []
    for i in range(len(labels)): # could put anything for len because the examples are always there as length
        # had two nested lists so have to do this in for loop to get one list of tuples
        predicted.append(tuple(zip(labels[i], probs_picked[i])))

    print(predicted[:20])

    # get the highest probability for sorting from all of the probabilities
    highest=0.0
    templist = []
    for i in range(len(probs)):
        for j in range(len(probs[i])):
            if probs[i][j] > highest:
                highest = probs[i][j]
        templist.append(highest)
        highest = 0.0


    # set label whether the text is toxic or clean
    pred_label = []
    for i in range(len(preds)):
        if sum(preds[i]) > 0:
            pred_label.append("toxic")
        else:
            pred_label.append("clean")

    label_prob = tuple(zip(pred_label, templist))

    print(label_prob[:5])

    all = tuple(zip(texts, prob_label_tuples, label_prob, predicted)) 
    #pprint(all[:10])

    # lists of tuples
    toxic = [item for item in all if item[2][0] == "toxic"]
    clean = [item for item in all if item[2][0] == "clean"]

    #now sort by probability, descending
    toxic.sort(key = lambda x: float(x[2][1]), reverse=True)
    clean.sort(key = lambda x: float(x[2][1]), reverse=True)

    all = [item for item in all]
    all.sort(key = lambda x: float(x[2][1]), reverse=True) # from most toxic to least toxic

    print("TOXIC")
    pprint(toxic[:10])
    print("NEUTRAL")
    pprint(toxic[-20:]) # these two in the middle are the most neutral things
    pprint(clean[:20])
    print("CLEAN")
    pprint(clean[-10:])

    # get the most toxic to tsv file
    toxicity  = [(toxic[i][0], toxic[i][2][0], toxic[i][2][1]) for i in range(len(toxic))]   
    cleaned  = [(clean[i][0], clean[i][2][0], clean[i][2][1]) for i in range(len(clean))]
    allpredict = [(all[i][0], all[i][2][0], all[i][2][1]) for i in range(len(all))]    

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

    probs = []
    for prob in probabilities:
        for p in prob:
            probs.append(p)

    # lastly use zip to get tuples with (text, label, probability)
    prediction_tuple = tuple(zip(texts, labels, probs))

    #pprint(prediction_tuple)

    toxic = [item for item in prediction_tuple
          if item[1] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1] == "clean"]

    # now sort by probability, descending
    toxic.sort(key = lambda x: float(x[2]), reverse=True)
    clean.sort(key = lambda x: float(x[2]))
    clean2 = sorted(clean, key = lambda x: float(x[2]), reverse=True)

    # beginning most toxic, middle "neutral", end most clean
    all = toxic + clean2

    print("TOXIC")
    pprint(toxic[:10]) # most toxic
    print("NEUTRAL")
    pprint(toxic[-10:]) # least toxic # this and the next is where the threshold can be seen and changed
    pprint(clean2[:10]) # "least clean"

    toxicity = toxic
    cleaned = clean
    allpredict = all


# get to dataframe
def text_and_label(data):
    df = pd.DataFrame(data, columns=['text', 'label', 'probability'])
    return df

# TODO here we are gonna lose all new line markers of the new dataset unless we make them straight into jsonl or something?
all_dataframe = text_and_label(allpredict)
all_dataframe = all_dataframe.replace(r'\n',' ', regex=True) # unix
all_dataframe = all_dataframe.replace(r'\r\n',' ', regex=True) # windows
all_dataframe = all_dataframe.replace(r'\r',' ', regex=True) # mac
all_dataframe.to_csv('predictions/all_predicted.tsv', sep="\t", header=False, index=False)

# get the most toxic to tsv file
dataframe = text_and_label(toxicity)
dataframe = dataframe.replace(r'\n',' ', regex=True) # unix
dataframe = dataframe.replace(r'\r\n',' ', regex=True) # windows
dataframe = dataframe.replace(r'\r',' ', regex=True) # mac

dataframe.to_csv('predictions/toxic_binary.tsv', sep="\t", header=False, index=False) 
dataframe2 = text_and_label(cleaned)
dataframe2 = dataframe2.replace(r'\n',' ', regex=True)
dataframe2 = dataframe2.replace(r'\r\n',' ', regex=True)
dataframe2 = dataframe2.replace(r'\r',' ', regex=True)
dataframe2.to_csv('predictions/clean_binary.tsv', sep="\t", header=False, index=False) 
