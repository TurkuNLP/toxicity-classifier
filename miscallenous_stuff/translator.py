from transformers import pipeline
import tqdm
import datetime
from transformers import MarianMTModel, MarianTokenizer
import sys
import pandas as pd
import json

print("start translation script")

# load the texts to be translated
data = sys.argv[1]
from ast import literal_eval
df = pd.read_csv(data, converters={'text': literal_eval}) # this works when the text column has lists, will fail if there are translations in there

num = 97311 # the number of rows translated previously and where to start (first row to take) TAKE NUM FROM PREVIOUS X ROWS TRANSLATED

texts = df["text"]
#print(texts[:5])

# take only the part that is translated for now
texts = texts.values
#print(texts[:5])

# instantiate the pipeline
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi", model_max_length=460) # added model max length
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=0) # , device=0
start = num
stop = 100
# go through every example (list of lists, so pipe gets list of sentences from one example)

def save_data(df, start, stop):
    # save to csv
    print()
    newdf = df.iloc[start:stop]
    newdf.to_csv('data/train-opus-mt-translated.csv', mode='a', index=False, header=False) # mode a appends to the file
    # save only the ones that were just translated (keep track of index) so do not have to redo stuff that much and resave the whole csv file every tim



print("beginning translation")

# all_texts = []
# for item in texts:
#     for one in texts:
#         all_texts.append(one)
# # this could then be turned into a dataset and the pipeline would be faster

# tr = pipe(all_texts, truncation=True, max_length=460) # translates everything at once
# translations = [t["translation_text"] for t in tr]
# with open("data/all_ids.txt", "r") as f:
#     indexes = f.readlines()

# indexes = [int(i) for i in indexes]

# begin = 0
# for one in indexes:
#     translation = translations[begin:one]
#     begin = one
#     final = ' '.join(translations)
#     df.at[num, 'text'] = final
#     stop = num + 1
#     if num % 10 == 0:
#         now = datetime.datetime.now()
#         print(now)
#         print(num+1, "rows translated")
#         # save to csv
#         save_data(df, start, stop)
#         start = i+num+1
#     num = num + 1

for i in tqdm.tqdm(range(len(texts[num:]))):
    tr = pipe(texts[i+num], truncation=True, max_length=460)
    #print(tr)
    translations = [t["translation_text"] for t in tr]
    # print("--")
    # print(translations)
    final = ' '.join(translations)
    #print(final)
    #print(i)
    
    # update the dataframe to have the translated text
    df.at[i+num, 'text'] = final
    #print(df["text"][i])

    stop = i+num+1
    current = i + num
    # save every 1000 rows
    if current % 10 == 0:
        now = datetime.datetime.now()
        print(now)
        print(i+num+1, "rows translated")
        # save to csv
        save_data(df, start, stop)
        start = i+num+1


print("all translated")
save_data(df, start, stop)



    # TODO make json file after translation
    # # what about saving to jsonl like the original files?
    # result = newdf.to_dict(orient="records")
    # print()
    # #parsed = json.loads(result)
    # with open('data/train-opus-mt-translated.json', 'a') as f: # a instead of f
    #     json.dump(result, f, separators=(',\n', ':'))

            # # what about saving to jsonl like the original files?
        # result = df.to_json(orient="records")
        # parsed = json.loads(result)
        # with open('data/opus-mt-translated.json', 'w') as f:
        #     json.dump(parsed, f)