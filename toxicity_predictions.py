import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

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


model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

trainer = transformers.Trainer(
    model=model
) 

data = args.data

# here set list of texts to use for new predictions
with open(data, 'r') as json_file:
        json_list = list(json_file)
lines = [json.loads(jline) for jline in json_list]
lines = lines[:200000] # CHANGE HOW MANY TEXTS WANT TO USE
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
        padding='max_length', # this got it to work...
        max_length=512,
        truncation=True,
    )

dataset = datasets.Dataset.from_pandas(df)

#pprint(dataset[:10])

#map all the examples
dataset = dataset.map(tokenize)

#pprint(dataset[:5])

texts = dataset["text"]

threshold = args.threshold

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions

# what to do with the predictions depending on the type
if args.type == "binary":
    import torch.nn.functional as F
    tensor = torch.from_numpy(predictions)
    probabilities = F.softmax(tensor, dim=1) # turn to probabilities using softmax

    print(probabilities[:10]) # this is now a tensor with two probabilities per example (two labels)
    print(predictions[:10])


    # THIS
    #preds = predictions.argmax(-1) # the -1 gives the indexes of the predictions, takes the one with the biggest number
     # argmax can be used on the probabilities as well although the tensor needs to changed to numpy array first

    # OR THIS
    # idea that if there is no high prediction for e.g. clean label then we set it to toxic (or the other way around)
    # set p[0] or p[1] depending on which we wanna concentrate on
    preds = [0 if p[1] < threshold else np.argmax(p) for p in predictions]  # if toxic below 0.5 count as clean
    # this technically messes up the probabilities when looking at the predictions but whatever lol
    # then I will have to look at the lower clean stuff as well because it will show there
    # TODO add some marking that a threshold was used?


    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    # now just loop to a list, get the probability with the indexes from preds
    probabilities = probabilities.tolist()
    probs = []
    for i in range(len(probabilities)):
        probs.append(probabilities[i][preds[i]]) # preds[i] gives the correct index for the current probability

    # lastly use zip to get tuples with (text, label, probability)
    prediction_tuple = tuple(zip(texts, labels, probs))

    # make into list of tuples
    toxic = [item for item in prediction_tuple
          if item[1] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1] == "clean"]

    # now sort by probability, descending
    toxic.sort(key = lambda x: float(x[2]), reverse=True)
    clean.sort(key = lambda x: float(x[2]), reverse=True)
    clean2 = sorted(clean, key = lambda x: float(x[2])) # ascending

    # beginning most toxic, middle "neutral", end most clean
    # all = toxic + clean2

    pprint(toxic[:10])
    pprint(toxic[-20:]) # these two middle are the closest to "neutral" where the threshold is
    pprint(clean[-20:])
    pprint(clean[:10])


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

    print(probs[:10])
    print(preds[:10])

    # put all the probabilities to a list with all the labels
    prob_label_tuples = []
    for prob in probs:
        prob_label_tuples.append(tuple(zip(prob, label_names[:-1])))


    pred_label_idxs = [] # the predicted indexes
    for vals in preds:
        pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

    # all the probabilities for the predicted labels
    probs_picked = []
    for i in range(len(probs)):
        if pred_label_idxs[i]:
            probs_picked.append([probs[i][val] for val in pred_label_idxs[i]])
        else:
            probs_picked.append(pred_label_idxs[i])


    labels = [] # the predicted labels
    idx2label = dict(zip(range(6), label_names[:-1]))   # could add clean
    for vals in pred_label_idxs:
        if vals:
            labels.append([idx2label[val] for val in vals])
        else:
            labels.append(vals)

    # the predicted labels and their probabilities
    predicted = list(zip(labels, probs_picked)) # this doesn't zip because list of lists? tensor related? TODO fix or don't
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

    # all.sort(key = lambda x: float(x[2][1]), reverse=True) # from most toxic to least toxic

    print("TOXIC")
    pprint(toxic[:10])
    print("NEUTRAL")
    pprint(toxic[-20:]) # these two in the middle are the most neutral things
    pprint(clean[:20])
    print("CLEAN")
    pprint(clean[-10:])

    # print("TOXIC")
    # pprint(toxic)
    # print("CLEAN")
    # pprint(clean)

    # pprint(all[:10])
    # pprint(all[-10:])

elif args.type == "true-binary":
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(predictions))
    print(predictions[:10])
    print(probabilities[:10])

    y_pred = [1 if prob >= threshold else 0 for prob in probabilities] 
    preds = y_pred

    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    # lastly use zip to get tuples with (text, label, probability)
    prediction_tuple = tuple(zip(texts, labels, probabilities))

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

    pprint(toxic[:10]) # most toxic
    pprint(toxic[-10:]) # least toxic # this and the next is where the threshold can be seen and changed
    pprint(clean2[:10]) # "least clean"




