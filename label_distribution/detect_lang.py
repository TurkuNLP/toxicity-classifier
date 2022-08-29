from langdetect import detect
from langdetect import DetectorFactory
import sys
import pandas as pd

DetectorFactory.seed = 0

# get the data to dataframe
data = sys.argv[1]

with open(data, 'r') as json_file:
        json_list = list(json_file)
lines = [json.loads(jline) for jline in json_list]

# use pandas to look at each column
df=pd.DataFrame(lines)
df = df[['body']]
df.rename(columns = {'body':'text'}, inplace = True) # have to change the column name so this works


# get texts and loop them to pick out the english texts
texts = df['text'].values.tolist()

en_list = []
for text in texts:
    if detect(text) == 'en':
        en_list.append(text)


# save the en_list to a file
with open('label_distribution/toxic_eng.txt', 'w') as f:
    for line in textlist:
        f.write(f"{line}\n")