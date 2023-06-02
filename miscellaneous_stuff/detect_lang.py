from langdetect import detect
from langdetect import DetectorFactory
import sys
import json
import pandas as pd

"""Script to see what languages are predicted to exist in the test data (or any data). Not the best language detector but does for this simple test."""


DetectorFactory.seed = 0

# get the data to dataframe
train = sys.argv[1]
test = sys.argv[2]

lines = []
def get_lines(data):
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    return lines

lines = lines + get_lines(train)
lines = lines + get_lines(test)

# use pandas to look at each column
df=pd.DataFrame(lines)
df = df[['text']]

# get texts and loop them to pick out the english texts
texts = df['text'].values.tolist()

lang_list = {}
count=0
for text in texts:
    try:
        lang = detect(text)
    except:
        print("no language detected?")
        lang = "no language detected"
    if lang in lang_list:
            count = lang_list.get(lang)
            lang_list[lang] = count + 1 # update the dictionary
            count = 0
    else:
        lang_list[lang] = 1 # add to the dictionary
        count = 0

print(lang_list)

# save the en_list to a file
with open('miscallenous_stuff/detected_langs', 'w') as f:
    for line in lang_list:
        f.write(f"{line}\n")