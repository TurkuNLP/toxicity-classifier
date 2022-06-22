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
parser.add_argument('--treshold', type=float, default=0.5,
    help="The treshold which to use for predictions, used in evaluation"
)
args = parser.parse_args()


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
    print(lines[:3])
    # there is now a list of dictionaries

    df=pd.DataFrame(lines)
    print(df.head())

    df['labels'] = list(df[label_names].values)
    print(df.head())

    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    print(df.head())

    set = datasets.Dataset.from_pandas(df)

    return set, df


train, unnecessary = json_to_dataset(args.train)
test, df = json_to_dataset(args.test)

# then split train into train and dev
train, dev = train.train_test_split(test_size=0.2).values()

# then make the dataset
dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})
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

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_names), problem_type="multi_label_classification")

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





#compute accuracy and loss
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels):
    # the treshold has to be really low because the probabilities of the predictions are not great, could even do without any treshold then? or find one that works best between 0.1 and 0.5
    threshold=args.treshold

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels) # why is the sigmoid applies? could do without it
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
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
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
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
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

trainer.train()



eval_results = trainer.evaluate(dataset["test"])
pprint(eval_results)
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
preds[np.where(probs >= args.treshold)] = 1

from sklearn.metrics import classification_report
print(classification_report(trues, preds, target_names=label_names))





# output dataframe to see what went wrong and what went right
# modified from https://gist.github.com/rap12391/ce872764fb927581e9d435e0decdc2df#file-output_df-ipynb
# COULD ALSO BE USED WITH REGISTER LABELING

idx2label = dict(zip(range(6), label_names))
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


