import datasets
import transformers
from pprint import PrettyPrinter
import logging
import argparse
import pandas as pd
import numpy as np
import json
import torch

# this should prevent any caching problems I might have because caching does not happen anymore
datasets.disable_caching()

pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)


parser = argparse.ArgumentParser(
        description="A script for classifying toxic data in a binary manner",
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
parser.add_argument('--dev', action='store_true', default=False,
    help="Decide whether to split the train into train and dev or not")
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

def json_to_dataset(data):
    # first I need to read the json lines
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # print(lines[:3])
    # there is now a list of dictionaries

    df=pd.DataFrame(lines)
    # print(df.head())
    df['labels'] = df[label_names].values.tolist()
    # print(df.head())

    # change to binary: if toxic 1 if clean 0
    # first get sum of labels
    df['labels'] = df.labels.map(sum) #df[label_names].sum(axis=1)

    # check that the ratio between clean and toxic is still the same! (it is)
    train_toxic = df[df["labels"] > 0]
    train_clean = df[df["labels"] == 0]
    print("toxic: ", len(train_toxic))
    print("clean: ", len(train_clean))

    # then change bigger than 0 to 1 and 0 stays 0
    df.loc[df["labels"] > 0, "labels"] = 1 #, "labels"

    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df)

    return dataset, df


train, traindf = json_to_dataset(args.train)
test, df = json_to_dataset(args.test)

# class weights
def class_weights(traindf):
    # get all labels from train
    labels = traindf["labels"].values.tolist()
    n_samples = (len(labels))
    n_classes = 2
    from collections import Counter
    c=Counter(labels)
    w1=n_samples / (n_classes * c[0])
    w2=n_samples / (n_classes * c[1])
    weights = [w1,w2]
    class_weights = torch.tensor(weights).to("cuda:0") # have to decide on a device
    print(class_weights)

class_weights = class_weights(traindf)

if args.dev == True:
    # then split train into train and dev
    train, dev = train.train_test_split(test_size=0.2).values()
    # then make the dataset
    dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})
else:
    dataset = datasets.DatasetDict({"train":train, "test":test})
print(dataset)


model_name = args.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True
    )
    
dataset = dataset.map(tokenize)

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir="../new_cache_dir/")

# Set training arguments 
trainer_args = transformers.TrainingArguments(
    "checkpoints",
    evaluation_strategy="epoch",
    logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    learning_rate=args.learning,
    #metric_for_best_model = "eval_f1", # this changes the best model to take the one with the best (biggest) f1 instead of best (smallest) TRAINING loss (NOT EVAL LOSS)
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32
)


# COMPUTE METRICS
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
def compute_metrics(pred):
    labels = pred.label_ids
    #print(labels)
    preds = pred.predictions.argmax(-1)
    #print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    wacc = balanced_accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'weighted_accuracy': wacc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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


# make own loss function with cross entropy loss
class newTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
            labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss



if args.dev == True:
    eval_dataset=dataset["dev"] 
else:
    eval_dataset=dataset["test"] #.select(range(20_000)) # just like topias does it

trainer = transformers.Trainer(
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
#print('weighted accuracy', eval_results['eval_weighted_accuracy'])

# see how the labels are predicted
test_pred = trainer.predict(dataset['test'])
trues = test_pred.label_ids
predictions = test_pred.predictions
preds = predictions.argmax(-1)


from sklearn.metrics import classification_report
print(classification_report(trues, preds, target_names=["clean", "toxic"]))