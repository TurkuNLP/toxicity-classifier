import datasets
import transformers
from pprint import PrettyPrinter
import logging
import argparse
import pandas as pd
import numpy as np
import json
import torch
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, classification_report
from collections import defaultdict

""" Toxicity classifier

This script is to be used for toxicity classification with jigsaw toxicity dataset in English (which is the original language)
 and Finnish (to which the data was translated using DeepL). The data is accepted in a .jsonl format and the data can be found in the data folder.

The labels of the dataset are:
- label_identity_attack
- label_insult
- label_obscene
- label_severe_toxicity
- label_threat
- label_toxicity
- label_clean
- no labels means that the text is clean


The script includes a binary classification where if there is a label for the text it is considered toxic and it there are no labels the text is clean/non-toxic.

List for necessary packages to be installed (could also check import list):
- pandas
- transformers
- datasets
- numpy
- torch

Information about the arguments to use with script can be found by looking at the argparse arguments with 'python3 toxic_classifier.py -h'.
"""


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
#usage as args.VARIABLENAME


label_names = [
    'label_identity_attack',
    'label_insult',
    'label_obscene',
    'label_severe_toxicity',
    'label_threat',
    'label_toxicity'
]

def json_to_dataset(data):
    """ Reads the data from .jsonl format and turns it into a dataset using pandas.
    
    Parameters
    ----------
    data: str
        path to the file from which to get the data

    Returns
    -------
    dataset: Dataset
        the data in dataset format
    """

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

    # check that the ratio between clean and toxic is still the same! (it is)
    train_toxic = df[df["labels"] > 0]
    train_clean = df[df["labels"] == 0]
    print("toxic: ", len(train_toxic))
    print("clean: ", len(train_clean))

    # then change bigger than 0 to 1 and 0 stays 0
    df.loc[df["labels"] > 0, "labels"] = 1

    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df)

    return dataset


def class_weights(train):
    """Calculates class weights for the loss function based on the train split. """

    labels = train["labels"] # get all labels from train
    n_samples = (len(labels))
    n_classes = 2
    c=Counter(labels)
    w1=n_samples / (n_classes * c[0])
    w2=n_samples / (n_classes * c[1])
    weights = [w1,w2]
    class_weights = torch.tensor(weights).to("cuda:0") # have to decide on a device

    print(class_weights)
    return class_weights


def compute_metrics(pred):
    """Computes the metrics"""

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

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


class newTrainer(transformers.Trainer):
    """Custom trainer to use class weights for loss"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computes loss with different class weights"""

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
            labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss



def main():
    # this should prevent any caching problems I might have because caching does not happen anymore
    datasets.disable_caching()

    pprint = PrettyPrinter(compact=True).pprint
    logging.disable(logging.INFO)

    train = json_to_dataset(args.train)
    test = json_to_dataset(args.test)

    class_weights = class_weights(train)

    if args.dev == True:
        # then split test into test and dev
        test, dev = test.train_test_split(test_size=0.2).values() # splitting shuffles by default
        train = train.shuffle(seed=42) # shuffle the train set
        # then make the dataset
        dataset = datasets.DatasetDict({"train":train,"dev":dev, "test":test})
    else:
        train = train.shuffle(seed=42) # shuffle the train set
        test = test.shuffle(seed=42) # shuffle the test set
        dataset = datasets.DatasetDict({"train":train, "test":test})
    print(dataset)

    #build model

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
        "checkpoints/binary",
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

    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5
    )
    training_logs = LogSavingCallback()


    # decide which eval set to use
    if args.dev == True:
        eval_dataset=dataset["dev"] 
    else:
        eval_dataset=dataset["test"] #.select(range(20_000))

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

    print(classification_report(trues, preds, target_names=["clean", "toxic"]))

if __name__ == "__main__":
    main()