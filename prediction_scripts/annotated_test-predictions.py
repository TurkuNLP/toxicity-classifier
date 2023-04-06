import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd
from transformers import EvalPrediction
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import hamming_loss

#python3 miscallenous_stuff/get_test-predictions.py --model "../models/finbert-large-deepl" --data "../data/test_fi_deepl.jsonl" --tokenizer "TurkuNLP/bert-base-finnish-cased-v1" --filename "test.tsv"

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
parser.add_argument('--new_test', action='store_true', default=False,
    help="use if testing and getting predictions for the new test set")
parser.add_argument('--threshold', type=float, default=None)
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint

if args.new_test == False:
    # read the data in
    data = args.data

    with open(data, 'r') as json_file:
            json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]

    print("number of lines in the file", len(lines))

    # use pandas to look at each column
    df=pd.DataFrame(lines)
    df['labels'] = df[label_names[:-1]].values.tolist()


if args.new_test == True:
# FOR OUR NEW TEST SET

    with open("../annotations/all_annotations.tsv", "rt", encoding="utf-8") as f:
        data = f.readlines()
    data = data[1:]
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t")
        assert len(data[i]) == 3

    df = pd.DataFrame(data, columns =['ID', 'label', 'text'])


    #change the annotation data to multi-label format

    # read the config.json file to see how many labels the model uses
    with open(f"{args.model}/config.json", 'r') as config_file:
        config_json = json.load(config_file)
    # here search for the correct part
    #print(len(config_json["id2label"])) # check length

    if len(config_json["id2label"]) == 6:
        # if the label includes not- something
        df.loc[df['label'].str.contains("not-"),["labels"]] = '[0,0,0,0,0,0]'

        # if threat
        df.loc[df["label"] == "threat",["labels"]] = '[0,0,0,0,1,0]'

        # if toxicity
        df.loc[df["label"] == "toxicity",["labels"]] = '[0,0,0,0,0,1]'

        #if severe_toxicity
        df.loc[df["label"] == "severe_toxicity",["labels"]] = '[0,0,0,1,0,0]'

        #if insult
        df.loc[df["label"] == "insult",["labels"]] = '[0,1,0,0,0,0]'

        #if identity_attack
        df.loc[df["label"] == "identity_attack",["labels"]] = '[1,0,0,0,0,0]'

        #if obscene
        df.loc[df["label"] == "obscene",["labels"]] = '[0,0,1,0,0,0]'

        import ast
        df['labels'] = df['labels'].apply(lambda row: ast.literal_eval(row)) 
        df.rename(columns = {'ID':'id'}, inplace = True)

    elif len(config_json["id2label"]) == 7:
        # if the label includes not- something
        df.loc[df['label'].str.contains("not-"),["labels"]] = '[0,0,0,0,0,0,1]'

        # if threat
        df.loc[df["label"] == "threat",["labels"]] = '[0,0,0,0,1,0,0]'

        # if toxicity
        df.loc[df["label"] == "toxicity",["labels"]] = '[0,0,0,0,0,1,0]'

        #if severe_toxicity
        df.loc[df["label"] == "severe_toxicity",["labels"]] = '[0,0,0,1,0,0,0]'

        #if insult
        df.loc[df["label"] == "insult",["labels"]] = '[0,1,0,0,0,0,0]'

        #if identity_attack
        df.loc[df["label"] == "identity_attack",["labels"]] = '[1,0,0,0,0,0,0]'

        #if obscene
        df.loc[df["label"] == "obscene",["labels"]] = '[0,0,1,0,0,0,0]'

        import ast
        df['labels'] = df['labels'].apply(lambda row: ast.literal_eval(row)) 
        df.rename(columns = {'ID':'id'}, inplace = True)

labels = df['labels'].values.tolist()
texts = df['text'].values.tolist()

df = df[['text', 'id', 'labels']]
pprint(df)


# instantiate model, this is pretty simple
model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding='max_length', # this got it to work, data_collator could have helped as well?
        max_length=512,
        truncation=True,
    )


def optimize_threshold(predictions, labels):
    """A method for getting the best threshold according to the best micro f1-score in case a threshold was not given. Made by Anna Salmela. Documentation by me.
    
    Parameters
    --------
    predictions
        the predictions for the labels
    labels: list
        the correct labels for the examples

    Returns
    ------
    best_f1_threshold: int
        the best threshold value, global value for all the labels
    """

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5 # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro') # this metric could be changed to something else
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold 


def multi_label_metrics(predictions, labels, threshold):
    """A method for measuring different metrics depending on what type of classification is to be done according to the arguments given to the script.
    
    Modified from https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/

    Parameters
    ---------
    predictions
        the predictions for the labels
    labels
        the correct labels
    threshold
        the threshold value to use to get the predicted labels

    Returns
    ------
    metrics
        a dictionary which includes the metrics to print out

    """

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels

    print("before change")
    print(y_pred[:30])
    if args.new_test == True:

        # TODO add here the check for 6 or 7 labels
        # for testing the new test set, set probabilities for labels other than the true label to 0.
        for i in range(len(y_pred)):
            if y_true[i][0] == 1 and y_pred[i][0] == 1:
                y_pred[i] = [1,0,0,0,0,0,0]
            if y_true[i][5] == 1 and y_pred[i][5] == 1:
                y_pred[i] = [0,0,0,0,0,1,0]
            if y_true[i][3] == 1 and y_pred[i][3] == 1:
                y_pred[i] = [0,0,0,1,0,0,0]
            if y_true[i][1] == 1 and y_pred[i][1] == 1:
                y_pred[i] = [0,1,0,0,0,0,0]
            if y_true[i][2] == 1 and y_pred[i][2] == 1:
                y_pred[i] = [0,0,1,0,0,0,0]
            if y_true[i][4] == 1 and y_pred[i][4] == 1:
                y_pred[i] = [0,0,0,0,1,0,0]
            else:
                y_pred[i] = [0,0,0,0,0,0,1]


        print(probs[:10])
        print(y_pred)


    # change to not take clean label into account when computing metrics (delete last prediction)
    # technically there can be a clean label and a toxic label at the same time but this just takes the toxic labels
    if len(y_pred[0]) == 7:
        new_pred, new_true, new_probs = [], [], []
        for i in range(len(y_pred)):
            new_pred.append(y_pred[i][:-1])
        for i in range(len(y_true)):
            new_true.append(y_true[i][:-1])
        for i in range(len(probs)):
            new_probs.append(y_true[i][:-1])
        y_true = new_true
        y_pred = new_pred
        probs = new_probs


    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    
    probs_roc_auc = roc_auc_score(y_true=y_true, y_score=probs)
    micro_roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average = 'micro') # micro or macro orr?
    macro_roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average = 'macro') # micro or macro orr?

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    from sklearn.metrics import hamming_loss
    hamming = hamming_loss(y_true, y_pred)
    #print("hamming loss", hamming)
    # hamming loss value ranges from 0 to 1. Lesser value of hamming loss indicates a better classifier.


    # return as dictionary
    metrics = {'f1': f1,
                'f1_macro': f1_macro,
                'precision': precision,
                'recall': recall,
                'probs_roc_auc': probs_roc_auc,
                'micro_roc_auc': micro_roc_auc,
                'macro_roc_auc': macro_roc_auc,
                'accuracy': accuracy,
                'hamming loss': hamming}

    print(probs[:30])
    print(y_true[:30])
    print(y_pred[:30])

    print(classification_report(y_true, y_pred, target_names=label_names[:-1], labels=list(range(6))))
    
    return metrics

def compute_metrics(p: EvalPrediction):
    """ Computes the metrics and calls threshold optimizer and multi-label metrics. """
    
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    if args.threshold == None:
        best_f1_th = optimize_threshold(preds, p.label_ids)
        threshold = best_f1_th
        print("Best threshold:", threshold)
    else:
        threshold = args.threshold
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids,
        threshold=threshold)
    return result

class MultilabelTrainer(transformers.Trainer):
    """A custom trainer to use a different loss and to use different class weights"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computes the loss and uses the class weights if --loss was used as an argument"""

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


trainer = MultilabelTrainer(
    model=model,
    compute_metrics=compute_metrics,
    tokenizer = tokenizer
)

dataset = datasets.Dataset.from_pandas(df)
print(dataset)

#map all the examples
dataset = dataset.map(tokenize)

ids = dataset["id"]

eval_results = trainer.evaluate(dataset)
pprint(eval_results)




# # see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions


sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
print(predictions[:10])
#probs = probs.tolist()

if args.threshold == None:
    threshold = optimize_threshold(probs, labels)
else:
    threshold = args.threshold


preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

print("before change")
print(preds[2000:2050])

# if args.new_test == True: #WHY DOES THIS NOT WORK? HUH?
#     print("doing the changes!")
for i in range(len(preds)):
    if labels[i][0] == 1 and preds[i][0] == 1:
        preds[i] = [1,0,0,0,0,0,0]
    if labels[i][5] == 1 and preds[i][5] == 1:
        preds[i] = [0,0,0,0,0,1,0]
    if labels[i][3] == 1 and preds[i][3] == 1:
        preds[i] = [0,0,0,1,0,0,0]
    if labels[i][1] == 1 and preds[i][1] == 1:
        preds[i] = [0,1,0,0,0,0,0]
    if labels[i][2] == 1 and preds[i][2] == 1:
        preds[i] = [0,0,1,0,0,0,0]
    if labels[i][4] == 1 and preds[i][4] == 1:
        preds[i] = [0,0,0,0,1,0,0]
    else:
        preds[i] = [0,0,0,0,0,0,1]

# change to only take 6 labels from the 7
if len(preds[0]) == 7:
    new_preds = []
    for i in range(len(preds)):
        new_preds.append(preds[i][:-1])
    preds = new_preds

if len(labels[0]) == 7:
    new_labels = []
    for i in range(len(labels)):
        new_labels.append(labels[i][:-1])
    labels = new_labels

print(probs[2000:2050])
print(labels[2000:2050])
print(preds[2000:2050])


temp_list = []
all_labels = []
for i in range(len(preds)):
    for j in range(len(preds[i])):
        if preds[i][j] == 1:
            temp_list.append(label_names[j])
    all_labels.append(temp_list)
    temp_list = [] # empty the list before next round

gold_labels = []
for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] == 1:
            temp_list.append(label_names[j])
    gold_labels.append(temp_list)
    temp_list = [] # empty the list before next round

# all_labels = []
# # set list to have label if the prob is bigger than the threshold
# temp_list = []
# for i in range(len(probs)):
#     for j in range(len(probs[i])):
#         if probs[i][j] > threshold:
#             temp_list.append(label_names[j])
#     all_labels.append(temp_list)
#     temp_list = [] # empty the list before next round

all = tuple(zip(ids, gold_labels, all_labels, texts))
#pprint(all[:10])

allpredict = [item for item in all]

# get to dataframe
def id_and_label(data):
    df = pd.DataFrame(data, columns=['id', 'gold_labels', 'predicted_labels', 'text'])
    return df

all_dataframe = id_and_label(allpredict)

# put to csv so we don't need any new lines taken out
filename = args.filename
all_dataframe.to_csv(filename, sep="\t", index=False) # added sep to make tsv
