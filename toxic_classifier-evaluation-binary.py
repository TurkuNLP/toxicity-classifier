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
]



def json_to_dataset(data):
    # first I need to read the json lines
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries
    df=pd.DataFrame(lines)
    df['labels'] = df[label_names].values.tolist()
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
    # print(y_true)
    # print(y_pred)

    # BINARY EVALUATION
    new_pred=[]
    new_true=[]
    for i in range(len(y_pred)):
        if y_pred[i].sum() > 0:
            new_pred.append(1)
        else:
            new_pred.append(0)
    for i in range(len(y_true)):
        if y_true[i].sum() > 0:
            new_true.append(1)
        else:
            new_true.append(0)


    precision, recall, f1, _ = precision_recall_fscore_support(y_true=new_true, y_pred=new_pred, average='binary')
    acc = accuracy_score(new_true, new_pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
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
print('F1_micro:', eval_results['eval_f1'])

