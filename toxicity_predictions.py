import transformers
import torch
import numpy as np


model=transformers.AutoModelForSequenceClassification.from_pretrained("models/multi-toxic")

trainer = transformers.Trainer(
    model=model
) 
    
# here set list of texts to use for new predictions
texts = 

# see how the labels are predicted
test_pred = trainer.predict(texts)
predictions = test_pred.predictions

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
# next, use threshold to turn them into integer predictions
preds = np.zeros(probs.shape)

# set threshold, this can be changed accordingly depending on the raw data predictions
threshold = 0.6

preds[np.where(probs >= threshold)] = 1

# take the clean label away from the metrics
new_pred = []
for i in range(len(preds)):
    new_pred.append(preds[i][:-1])
preds = new_pred


print(probs)
print(preds)

# put the probabilities to a list with all the labels
prob_label_tuples = []
probs = probs.tolist()
for prob in probs:
    prob_label_tuples.append(tuple(zip(prob, label_names[:-1])))

# get indexes from the preds to get the probabilities for the predicted labels
probs_picked = []
for i in range(len(preds)):
    temp = []
    for j in range(len(pred[i])):
        if pred[i][j] == 1:
            temp.append(probs[i][j])
    if not temp: 
        # list is empty, still add it (although here it would make sense to have the clean label or just simply get all probabilities with their labels
        prob_label_idxs.append(temp)

    prob_label_idxs.append(temp) # here is then probability for every label appearing in the prediction

for vals in preds:
    pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

labels = []
idx2label = dict(zip(range(6), label_names[:-1]))   # could add clean
for vals in pred_label_idxs:
    if vals:
        labels.append([idx2label[val] for val in vals])
    else:
        labels.append(vals)

# the predicted labels and their probabilities
predicted = tuple(zip(labels, probs_picked))

# get the highest probability for sorting
highest=0
templist = []
for i in range(len(predicted)):
    for j in range(len(predicted[i])):
        if predicted[i][2][j] > highest: # the [2] can be empty though so I don't know if this will work, need a check for whether it's empty?
            highest = predicted[i][2][j]
    templist.append(highest)
    highest = 0


# # set label whether the text is toxic or clean
# pred_label = []
# for i in range(len(preds)):
#     if sum(preds[i]) > 0:
#         pred_label.append("toxic")
#     else:
#         pred_label.append("clean")

# label_prob = tuple(zip(pred_label, templist))


all = tuple(zip(texts, prob_label_tuples, templist, predicted)) 
pprint(all[:10])

# lists of tuples
# toxic = [item for item in all if item[2][1] == "toxic"]
# clean = [item for item in all if item[2][1] == "clean"]

# now sort by probability, descending
# toxic.sort(key = lambda x: float(x[2][2]), reverse=True)
# clean.sort(key = lambda x: float(x[2][2]), reverse=True)
#clean2 = sorted(clean, key = lambda x: float(x[2])) # ascending

all.sort(key = lambda x: float(x[2]), reverse=True)

# pprint(toxic[:5])
# pprint(clean[:5])

pprint(all[:5])
pprint(all[-5:])


