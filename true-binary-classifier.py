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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from collections import defaultdict
import matplotlib.pyplot as plt

""" Toxicity classifier

This script is to be used for toxicity classification with jigsaw toxicity dataset in English (which is the original language)
 and Finnish (to which the data was translated using DeepL). The data is accepted in a .jsonl format and the data can be found in the data folder of the repository.

The labels of the dataset are:
- label_identity_attack
- label_insult
- label_obscene
- label_severe_toxicity
- label_threat
- label_toxicity
- label_clean
+ no labels means that the text is clean


The script includes a binary classification where if there is a label for the text it is considered toxic and if there are no labels the text is clean/non-toxic.

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
parser.add_argument('--loss', action='store_true', default=False,
        help="If used different class weights are used for the loss function")
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
    df['labels'] = torch.tensor(df['labels'], dtype=torch.float) # this works but somehow there is still an error
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df)
    return dataset


def compute_metrics(pred):
    """Computes the metrics"""

    labels = pred.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.predictions))
    print(pred.predictions[:5])
    print(probs[:5])

    # change the threshold here! only one prediction where we decide which is which (default 0 if < 0.5 and 1 if >= 0.5)
    threshold = 0.6
    y_pred = [1 if prob >= threshold else 0 for prob in probs] 
    preds = y_pred

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary') # micro or binary??
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(y_true=labels, y_score=preds, average = 'micro')
    wacc = balanced_accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'weighted_accuracy': wacc,
        'roc_auc': roc_auc,
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


def predictions_to_csv(trues, preds, dataset):
    """ Saves a dataframe to .csv with texts, correct labels and predicted labels to see what went right and what went wrong.
    
    Modified from https://gist.github.com/rap12391/ce872764fb927581e9d435e0decdc2df#file-output_df-ipynb

    Parameters
    ---------
    trues: list
        list of correct labels
    preds: list
        list of predicted labels
    dataset: Dataset
        the dataset from which to get texts

    """

    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    print(idx2label)

    # Gathering vectors of label names using idx2label (modified single-label version)
    true_labels, pred_labels = [], []
    for val in trues:
        true_labels.append(idx2label[val])
    for val in preds:
        pred_labels.append(idx2label[val])

    #get the test texts to a list of their own 
    texts = dataset["test"]["text"]

    # Converting lists to df
    comparisons_df = pd.DataFrame({'text': texts, 'true_labels': true_labels, 'pred_labels':pred_labels})
    comparisons_df.to_csv('binary_comparisons.csv')
    #print(comparisons_df.head())



def get_predictions(dataset, trainer, pprint):
    test_pred = trainer.predict(dataset['test'])
    # this actually has metrics because the labels are available so evaluating is technically unnecessary since this does both! (checked documentation)

    predictions = test_pred.predictions # logits

    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(predictions))
    print(predictions[:5])
    print(probabilities[:5])

    # change the threshold here! only one prediction where we decide which is which (default 0 if < 0.5 and 1 if >= 0.5)
    threshold = 0.6
    y_pred = [1 if prob >= threshold else 0 for prob in probabilities] 
    preds = y_pred


    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    texts = dataset["test"]["text"]
    # lastly use zip to get tuples with (text, label, probability)
    prediction_tuple = tuple(zip(texts, labels, probabilities))

    #pprint(prediction_tuple)

    toxic = [item for item in prediction_tuple
          if item[1] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1] == "clean"]

    # now sort by probability, descending
    toxic.sort(key = lambda x: float(x[2]), reverse=True)
    clean.sort(key = lambda x: float(x[2]), reverse=True)
    clean2 = sorted(clean, key = lambda x: float(x[2])) # ascending

    # beginning most toxic, middle "neutral", end most clean
    all = toxic + clean2

    pprint(toxic[:5])
    pprint(toxic[-5:])
    pprint(clean[:5])



def main():
    # this should prevent any caching problems I might have because caching does not happen anymore
    datasets.disable_caching()

    pprint = PrettyPrinter(compact=True).pprint
    logging.disable(logging.INFO)

    train = json_to_dataset(args.train)
    test = json_to_dataset(args.test)

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

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, cache_dir="../new_cache_dir/")

    # Set training arguments 
    trainer_args = transformers.TrainingArguments(
        "checkpoints/binarytransfer",
        evaluation_strategy="epoch",
        logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning,
        #metric_for_best_model = "eval_f1", # this changes the best model to take the one with the best (biggest) f1 instead of the default: 
        #best (smallest) training or eval loss (seems to be random?)
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

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    # change the threshold here
    threshold = 0.6
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    preds = y_pred




    print(classification_report(trues, preds, target_names=["clean", "toxic"]))

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(trues, preds)

    #create precision recall curve using matplotlib
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='red')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    plt.show()
    plt.savefig("binary_precision-recall-curve") # set file name where to save the plots

    get_predictions(dataset, trainer, pprint)
    #predictions_to_csv(trues, preds, dataset)

if __name__ == "__main__":
    main()