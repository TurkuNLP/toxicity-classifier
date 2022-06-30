import datasets
import transformers
from pprint import PrettyPrinter
import logging
import argparse
import pandas as pd
import numpy as np
import json
import torch


pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)


parser = argparse.ArgumentParser(
        description="A script for classifying toxic data (multilabel)",
        epilog="Made by Anni Eskelinen"
    )
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--batch', type=int, default=8,
    help="The batch size for the model"
)
parser.add_argument('--epochs', type=int, default=3,
    help="The number of epochs to train for"
)
parser.add_argument('--learning', type=float, default=8e-6,
    help="The learning rate for the model"
)
parser.add_argument('--threshold', type=float, default=None,
    help="The treshold which to use for predictions, used in evaluation"
)
parser.add_argument('--loss', action='store_true', default=False,
    help="Decide whether to use the loss function or not")
parser.add_argument('--dev', action='store_true', default=False,
    help="Decide whether to split the train into train and dev or not")
args = parser.parse_args()

print(args)

# this I could instead parse from the data, now I got it manually
label_names = [
    'label_identity_attack',
    'label_insult',
    'label_obscene',
    'label_severe_toxicity',
    'label_threat',
    'label_toxicity',
    'label_clean' # added new label for clean ones so that I can try taking the clean data into account with the class weights (no labels)
]



def json_to_dataset(data):
    # first I need to read the json lines
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries
    df=pd.DataFrame(lines)
    df['labels'] = df[label_names[:-1]].values.tolist() # don't take clean label into account because it doesn't exist yet


    # TODO make argparse argument for this so I don't have to come here to use this everytime
    # add new column for clean data
    df['sum'] = df.labels.map(sum) 
    df.loc[df["sum"] > 0, "label_clean"] = 0
    df.loc[df["sum"] == 0, "label_clean"] = 1
    df['labels'] = df[label_names].values.tolist() # update labels column to include clean data



    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    dataset = datasets.Dataset.from_pandas(df)

    return dataset, df


train, unnecessary = json_to_dataset(args.train)
test, df = json_to_dataset(args.test)



# class weigths for the loss function
#implemented from https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c

labels = unnecessary["labels"].values.tolist() # get all labels from train data
n_samples = len(labels) # number of examples in train data
n_classes = len(label_names)

# Count each class frequency
class_count = [0] * n_classes
for classes in labels: # go through every label list (example)
    for index in range(n_classes):
        if classes[index] != 0:
            class_count[index] += 1

# Compute class weights using balanced method
class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
class_weights = torch.tensor(class_weights).to("cuda:0") # have to decide on a device
# multiply things if there is more than one
# does this help at all when the problem is with examples that have no labels?
# I believe this is somewhat based on scikit learns compute_class_weight method (which does not work for one hot encdded labels)
print(class_weights)

if args.dev == True:
    # then split train into train and dev
    train, dev = train.train_test_split(test_size=0.2).values()
    # then make the dataset
    dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})
else:
    dataset = datasets.DatasetDict({"train":train, "test":test})
print(dataset)


model_name = args.model # finbert for Finnish and bert for english? xlmr-base or large also
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True
    )
    
dataset = dataset.map(tokenize)

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_names), problem_type="multi_label_classification", cache_dir="../new_cache_dir/")

# Set training arguments CHANGE TO EPOCHS
trainer_args = transformers.TrainingArguments(
    "checkpoints",
    evaluation_strategy="epoch",
    logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    learning_rate=args.learning,
    #metric_for_best_model = "eval_f1", # this changes the best model to take the one with the best (biggest) f1 instead of best (smallest) loss
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32
)


# APPLYING TRESHOLD OPTIMIZATION COULD HELP WITH THE IMBALANCE ISSUE

# in case a threshold was not given, choose the one that works best with the evaluated data
def optimize_threshold(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5 # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro') # change this to weighted?
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold 


#compute accuracy and loss
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels) # why is the sigmoid applies? could do without it
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels


    # CHANGE TO NOT TAKE THE CLEAN LABEL INTO ACCOUNT WHEN COMPUTING THE SCORE
    new_pred, new_true = [], []    
    for i in range(len(y_pred)):
        new_pred.append(y_pred[i][:-1])
    for i in range(len(y_true)):
        new_true.append(y_true[i][:-1])


    f1_micro_average = f1_score(y_true=new_true, y_pred=new_pred, average='micro')
    f1_weighted_average = f1_score(y_true=new_true, y_pred=new_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
                'f1_weighted': f1_weighted_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
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

data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Argument gives the number of steps of patience before early stopping
early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5
)

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()


class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if args.loss == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

if args.dev == True:
    eval_dataset=dataset["dev"] 
else:
    eval_dataset=dataset["test"] #.select(range(20_000)) # just like topias does it

trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

trainer.train()



eval_results = trainer.evaluate(dataset["test"]) #.select(range(20_000)))
#pprint(eval_results)
print('F1_micro:', eval_results['eval_f1_micro'])
print('F1_weighted:', eval_results['eval_f1_weighted'])


# see how the labels are predicted
test_pred = trainer.predict(dataset['test'])
trues = test_pred.label_ids
predictions = test_pred.predictions

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
# next, use threshold to turn them into integer predictions
preds = np.zeros(probs.shape)
preds[np.where(probs >= args.threshold)] = 1

new_pred, new_true = [], []
for i in range(len(preds)):
    new_pred.append(preds[i][:-1])
for i in range(len(trues)):
    new_true.append(trues[i][:-1])

from sklearn.metrics import classification_report
print(classification_report(trues, preds, target_names=label_names[:-1], labels=list(range(6))))





# output dataframe to see what went wrong and what went right
# modified from https://gist.github.com/rap12391/ce872764fb927581e9d435e0decdc2df#file-output_df-ipynb
# COULD ALSO BE USED WITH REGISTER LABELING

idx2label = dict(zip(range(6), label_names[:-1]))
print(idx2label)

# Getting indices of where boolean one hot vector true_bools is True so we can use idx2label to gather label names
true_label_idxs, pred_label_idxs=[],[]
for vals in trues:
  true_label_idxs.append(np.where(vals)[0].flatten().tolist())
for vals in preds:
  pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

# Gathering vectors of label names using idx2label
true_label_texts, pred_label_texts = [], []
for vals in true_label_idxs:
  if vals:
    true_label_texts.append([idx2label[val] for val in vals])
  else:
    true_label_texts.append(vals)

for vals in pred_label_idxs:
  if vals:
    pred_label_texts.append([idx2label[val] for val in vals])
  else:
    pred_label_texts.append(vals)

#get the test texts to a list of their own 
texts = df[["text"]]
# turn it into an actual list
texts = texts.values.tolist()



# Converting lists to df
comparisons_df = pd.DataFrame({'text': texts, 'true_labels': true_label_texts, 'pred_labels':pred_label_texts})
comparisons_df.to_csv('comparisons.csv')
comparisons_df.head()


