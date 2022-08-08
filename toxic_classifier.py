import datasets
import transformers
from pprint import PrettyPrinter
import logging
import argparse
import pandas as pd
import numpy as np
import json
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
from collections import defaultdict
from transformers import EvalPrediction

""" Toxicity classifier

This script is to be used for toxicity classification with jigsaw toxicity dataset in English (which is the original language)
 and Finnish (to which the data was translated using DeepL).The data is accepted in a .jsonl format and the data can be found in the data folder of the repository.

The labels of the dataset are:
- label_identity_attack
- label_insult
- label_obscene
- label_severe_toxicity
- label_threat
- label_toxicity
- label_clean
+ no labels means that the text is clean


The script includes mainly four use cases:
- use the original intended way of the data: multi-label classification with the original labels
- adding the clean (no-labels) as a label (= 7 labels) of it's own to get class weights for the loss function 
and then evaluating the performance with the original 6 labels
- using the 6 labels for classification but evaluating it in a binary manner (toxic, clean/non-toxic)
- using the 7 labels for classification and evaluating in a binary manner

List for necessary packages to be installed (could also check import list): 
- pandas
- transformers
- datasets
- numpy
- torch

Information about the arguments to use with script can be found by looking at the argparse arguments with 'python3 toxic_classifier.py -h'.
"""

def arguments():
    """Uses argparser to get "optional" arguments from the command line. Use 'python3 toxic_classifier.py -h' to get information about them.

    Returns
    ------
    args: Namespace(objects)
        arguments as objects of a namespace, usage as args.VARIABLENAME
    """

    parser = argparse.ArgumentParser(
            description="A script for classifying toxic data (multi-label, includes binary evaluation on top)",
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
        help="The treshold which to use for predictions, used in evaluation. If no threshold is given threshold optimization is used."
    )
    parser.add_argument('--loss', action='store_true', default=False,
        help="If used different class weights are used for the loss function")
    parser.add_argument('--dev', action='store_true', default=False,
        help="If used the train set is split into train and dev sets")
    parser.add_argument('--clean_as_label', action='store_true', default=False,
        help="If used the clean examples (no label) are marked as having a label instead for class weights purposes")
    parser.add_argument('--binary', action='store_true', default=False,
        help="If used the evaluation uses a binary classification (toxic or not) based on the multi-label predictions.")
    args = parser.parse_args()

    print(args)
    return args

# keep arguments out of main method to keep it as a global variable
# get commandline arguments
#args = arguments()


def json_to_dataset(data, label_names):
    """ Reads the data from .jsonl format and turns it into a dataset using pandas.
    
    Parameters
    ----------
    data: str
        path to the file from which to get the data
    label_names: list
        list of the labels

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
    df['labels'] = df[label_names[:-1]].values.tolist() # don't take clean label into account because it doesn't exist yet


    if args.clean_as_label == True:
        # add new column for clean data
        df['sum'] = df.labels.map(sum) 
        # if there are no labels it is clean 1, and if there are none it is 0
        df.loc[df["sum"] > 0, "label_clean"] = 0
        df.loc[df["sum"] == 0, "label_clean"] = 1
        df['labels'] = df[label_names].values.tolist() # update labels column to include clean data


    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    dataset = datasets.Dataset.from_pandas(df)

    return dataset


def make_class_weights(train, label_names):
    """Calculates class weights for the loss function based on the train split.

    implemented from https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c
    based on scikit learns compute_class_weight method (which does not work for one hot encoded labels)
    
    Parameters
    ---------
    train: Dataset
        train split of the dataset
    label_names: list
        list of the labels used in the data

    Returns
    ------
    class_weights: list
        a list of the class weights

    """

    labels = train["labels"] # get all rows (examples) from train data, labels only
    n_samples = len(labels) # number of examples (rows) in train data
    # number of unique labels
    if args.clean_as_label == True:
        n_classes = len(label_names)
    else:
        n_classes = len(label_names) -1

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in labels: # go through every label list (example)
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_weights = torch.tensor(class_weights).to("cuda:0") # have to decide on a device
    print(class_weights)

    return class_weights

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
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro') # this metric could be changed to something else
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


    if args.clean_as_label == True:
        # change to not take clean label into account when computing metrics (delete last prediction)
        # technically there can be a clean label and a toxic label at the same time but this just takes the toxic labels
        new_pred, new_true = [], []    
        for i in range(len(y_pred)):
            new_pred.append(y_pred[i][:-1])
        for i in range(len(y_true)):
            new_true.append(y_true[i][:-1])
        y_true = new_true
        y_pred = new_pred

    if args.binary == True:
        # binary evaluation
        # if there are labels the text is toxic = 1
        new_pred, new_true = [], []
        for i in range(len(y_pred)):
            if sum(y_pred[i]) > 0:
                new_pred.append(1)
            else:
                new_pred.append(0)
        for i in range(len(y_true)):
            if sum(y_true[i]) > 0:
                new_true.append(1)
            else:
                new_true.append(0)
        y_true = new_true
        y_pred = new_pred

        precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary')
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average = 'micro')
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')
        f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average = 'micro')
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        # return as dictionary
        metrics = {'f1': f1,
                    'f1_weighted': f1_weighted_average,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'accuracy': accuracy}
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


class MultilabelTrainer(transformers.Trainer):
    """A custom trainer to use a different loss and to use different class weights"""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computes the loss and uses the class weights if --loss was used as an argument"""

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if args.loss == True:
            # include class weights in loss computing
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight= self.class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def get_classification_report(trainer, label_names, dataset):
    """Prints the classification report for the predictions. Different reports based on whether using binary evaluation or multi-label classification.
    
    Parameters
    --------
    trainer: Trainer
        the fully trained model used to make predictions
    label_names: list
        list of the labels used in the data
    dataset: Dataset
        the dataset from which to predict the labels

    Returns
    ------
    trues
        list of the correct labels
    preds
        list of the predicted labels
    """

    # see how the labels are predicted
    test_pred = trainer.predict(dataset['test'])
    trues = test_pred.label_ids
    predictions = test_pred.predictions

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    preds = np.zeros(probs.shape)
    if args.threshold == None:
        best_f1_th = optimize_threshold(preds, trues)
        threshold = best_f1_th
        print("Best threshold:", threshold)
    else:
        threshold = args.threshold
    preds[np.where(probs >= threshold)] = 1

    # take the clean label away from the metrics
    if args.clean_as_label == True:
        new_pred, new_true = [], []
        for i in range(len(preds)):
            new_pred.append(preds[i][:-1])
        for i in range(len(trues)):
            new_true.append(trues[i][:-1])
        trues = new_true
        preds = new_pred
    
    if args.binary == True:
        # binary evaluation
        # if there are labels the text is toxic = 1
        new_pred, new_true = [], []
        for i in range(len(preds)):
            if sum(preds[i]) > 0:
                new_pred.append(1)
            else:
                new_pred.append(0)
        for i in range(len(trues)):
            if sum(trues[i]) > 0:
                new_true.append(1)
            else:
                new_true.append(0)
        print(classification_report(new_true, new_pred, target_names=["clean", "toxic"], labels=list(range(2))))

    # this report shows up even with binary evaluation but I don't think it matters, good info nonetheless
    print(classification_report(trues, preds, target_names=label_names[:-1], labels=list(range(6))))

    return trues, preds


def predictions_to_csv(trues, preds, dataset, label_names):
    """ Saves a dataframe to .csv with texts, correct labels and predicted labels to see what went right and what went wrong.
    
    Modified from https://gist.github.com/rap12391/ce872764fb927581e9d435e0decdc2df#file-output_df-ipynb

    Parameters
    ---------
    trues: list
        list of correct labels
    preds: list
        list of predicted labels
    dataset: Dataset
        the dataset from which to get the texts
    label_names: list
        list of the labels used in the data
    """

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
    texts = dataset["test"]["text"]
    print(len(texts), len(true_label_texts), len(pred_label_texts))

    # Converting lists to df
    comparisons_df = pd.DataFrame({'text': texts, 'true_labels': true_label_texts, 'pred_labels':pred_label_texts})
    comparisons_df.to_csv('comparisons.csv')
    #print(comparisons_df.head())


def main():
    # this should prevent any caching problems I might have because caching does not happen anymore
    datasets.disable_caching()

    pprint = PrettyPrinter(compact=True).pprint
    logging.disable(logging.INFO)

    label_names = [
        'label_identity_attack',
        'label_insult',
        'label_obscene',
        'label_severe_toxicity',
        'label_threat',
        'label_toxicity',
        'label_clean' # added new label for clean examples (no label previously) because the no label ones were not taken into account in the class weights
    ]

    # data to dataset format
    train= json_to_dataset(args.train, label_names)
    test = json_to_dataset(args.test, label_names)

    # use loss if so specified
    if args.loss == True:
        class_weights = make_class_weights(train, label_names)

    # use dev set or not
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


    # build the model

    model_name = args.model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            max_length=512,
            truncation=True
        )
        
    dataset = dataset.map(tokenize)

    # number of labels
    if args.clean_as_label == True: 
        num_labels=len(label_names)
    else:
        num_labels=len(label_names) - 1
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification", cache_dir="../new_cache_dir/")

    # Set training arguments
    trainer_args = transformers.TrainingArguments(
        "checkpoints/multilabel", #output_dir for checkpoints, not necessary to mention what it is
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

    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5
    )
    training_logs = LogSavingCallback()

    # which evaluation set to use
    if args.dev == True:
        eval_dataset=dataset["dev"] 
    else:
        eval_dataset=dataset["test"]

    trainer = MultilabelTrainer(
        class_weights = class_weights,
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


    eval_results = trainer.evaluate(dataset["test"])
    #pprint(eval_results)
    print('F1:', eval_results['eval_f1'])

    trues, preds = get_classification_report(trainer, label_names, dataset)

    if args.binary == False:
        predictions_to_csv(trues, preds, dataset, label_names)


if __name__ == "__main__":
    main()
