from transformers import pipeline
import datetime
from transformers import MarianMTModel, MarianTokenizer
import sys
import pandas as pd
import json

# load the texts to be translated
data = sys.argv[1]
from ast import literal_eval
df = pd.read_csv(data, converters={'text': literal_eval}) # oh this does not work because the whole file isn't lists yet , converters={'text': literal_eval}
texts = df["text"]
print(texts[:5])

# take only the part that is translated for now
texts = texts.values
print(texts[:5])

# instantiate the pipeline
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi", model_max_length=460) # added model max length
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=0)

def save_data(df):
    # save to csv
    df.to_csv('data/train-opus-mt-translated.csv', index=False)

    # what about saving to jsonl like the original files?
    result = df.to_dict(orient="records")
    parsed = json.loads(result)
    with open('data/train-opus-mt-translated.json', 'w') as f:
        json.dump(parsed, f)


# go through every example (list of lists, so pipe gets list of sentences from one example)
for i in range(len(texts)):
    tr = pipe(texts[i], truncation=True, max_length=512)
    #print(tr)
    translations = [t["translation_text"] for t in tr]
    # print("--")
    # print(translations)
    final = ' '.join(translations)
    print(final)
    
    # update the dataframe to have the translated text
    df.at[i, 'text'] = final
    #print(df["text"][i])

    # save every 1000 rows
    if i % 100 == 0:
        now = datetime.datetime.now()
        print(now)
        print(i+1, "rows translated")
        # save to csv
        save_data(df)

        # what about saving to jsonl like the original files?
        result = df.to_json(orient="records")
        parsed = json.loads(result)
        with open('data/opus-mt-translated.json', 'w') as f:
            json.dump(parsed, f)


print("all translated")
save_data(df)

        # tokenized_input = tokenizer.tokenize(newstr)
        # if len(tokenized_input) > 400:
        #     print(i, newstr)