from transformers import pipeline
import tqdm
import datetime
from transformers import MarianMTModel, MarianTokenizer
import sys
import pandas as pd
import json

print("start translation script")

begin = 0 # the number of rows translated previously
data = sys.argv[1]

# if using the csv files done previously!

# # load the texts to be translated
# from ast import literal_eval
# df = pd.read_csv(data, converters={'text': literal_eval}) # this works when the text column has lists, will fail if there are translations in there
# texts = df["text"]
# #print(texts[:5])

# # take only the part that is translated for now
# texts = texts.values
#print(texts[:5])

# else just use this
import datasets
dataset = dataset = datasets.load_dataset("json", data_files=data)
print(dataset)

texts = dataset["train"]['text'] # automatically loads as train split in the above

# instantiate the pipeline
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi", model_max_length=460) # added model max length
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=0) # , device=0
# go through every example (list of lists, so pipe gets list of sentences from one example)

def save_data(comments, current):
    # here take the original data file train_en.jsonl and only change the text fields in each and save them
    all_dictionaries = []
    previous = current - 100

    with jsonlines.open('data/split_output_en.jsonl') as reader:
        for obj in reader:
            num = previous
            print(obj)
            dictionary = obj
            # change the text in each to new one
            dictionary["text"] = comments[num] 
            all_dictionaries.append(dictionary)
            if num + 1 > current:
                break
            else:
                num += 1

    with jsonlines.open('data/translated_en-fi.jsonl', mode='a') as writer:
        for item in all_dictionaries:
            writer.write(item)


print("beginning translation")
comments = []

for i in tqdm.tqdm(range(len(texts[begin:]))):
    # what was my reason for max_length 460?
    tr = pipe(texts, truncation=True, max_length=460) # this should only do the texts for translation, can also just use 'texts' instead if this does not work for some reason
    #print(tr)
    translations = [t["translation_text"] for t in tr]
    # print("--")
    # print(translations)
    final = ' '.join(translations)
    #print(final)
    #print(i)
    
    # make list of final texts
    comments.append(final)

    # save every 1000 rows
    if i % 100 == 0 and i != 0:
        print(datetime.datetime.now())
        print(i+begin+1, "rows translated")
        current = 1
        save_data(comments, current)


print("all translated")
save_data(final)