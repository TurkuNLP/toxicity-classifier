from transformers import pipeline
import tqdm
import datetime
from transformers import MarianMTModel, MarianTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import sys
import pandas as pd
import json

"""Script for translating text using opus-mt, the text has been previously split to sentences. Save as jsonl. """


print("start translation script")

begin = 0 # the number of rows translated previously
data = sys.argv[1]
og = sys.argv[2]

# if using the csv files done previously!

# # load the texts to be translated
# from ast import literal_eval
# df = pd.read_csv(data, converters={'text': literal_eval}) # this works when the text column has lists, will fail if there are translations in there
# texts = df["text"]
# #print(texts[:5])

# # take only the part that is translated for now
# texts = texts.values
#print(texts[:5])

#get the split texts for translation
split_texts = []
with open(data, 'r', encoding='utf-8') as f:
    for line in f:
        split_texts.append(json.loads(line))


# instantiate the pipeline
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi", model_max_length=460) # added model max length
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=0) # , device=0
# go through every example (list of lists, so pipe gets list of sentences from one example)

# get the original file with eng text and labels for processing the data for saving 
with open(og, 'r') as json_file:
    json_list = list(json_file)
    all_jsonlines = [json.loads(jline) for jline in json_list] # list of dictionaries

def save_data(comments, current, previous):
    # here take the original data file train_en.jsonl and only change the text fields in each and save them
    all_dictionaries = []
    num = previous
    for obj in all_jsonlines[previous:]:
        dictionary = obj
        # change the text in each to new one
        dictionary["text"] = comments[num] 
        #print(comments[num])
        all_dictionaries.append(dictionary)
        num += 1
        if num > current:
            break

    with jsonlines.open('data/new_opus_translations/translated_en-fi.jsonl', mode='a') as writer:
        for item in all_dictionaries:
            writer.write(item)


print("beginning translation")
comments = []
previous = begin

for i in tqdm.tqdm(range(len(split_texts[begin:]))): 
    text = split_texts[i+begin]["text"]
    tr = pipe(text, truncation='only_first', max_length=460) # what was my reason for max_length 460?
    #print(tr)
    translations = [t["translation_text"] for t in tr] # tr[0]["translation_text"] 
    # print("--")
    #print(translations)
    final = ' '.join(translations)
    #print(final)
    #print(final)
    #print(i)
    
    # make list of final texts
    comments.append(final)

    # save every 100 rows
    if i % 100 == 0 and i != 0:
        print(datetime.datetime.now())
        print(i+begin+1, "rows translated")
        current = i + begin
        save_data(comments, current, previous)
        previous = i + begin


print("all translated")
save_data(comments, len(split_texts) - 1, previous) # this goes through all the comments and the texts to make the final file