import json
import jsonlines
import sys
import datasets

data = sys.argv[1]

dataset = datasets.load_dataset("json", data_files=data)
print(dataset)

texts = dataset["train"]['text'] # automatically loads as train split in the above
ids = dataset["train"]['id']
final = []

thusfar = 0 # the line number where to continue if the pipe fails or something

# TODO make this work :D remove everything related the url stuff
import ufal.udpipe
udpipemodel=ufal.udpipe.Model.load("udpipe/english-ewt.udpipe") # change to a finnish model for backtranslation
tokenizer=ufal.udpipe.Pipeline(udpipemodel,"tokenizer=ranges","none","none","conllu")
for i in range(len(texts)):
    parsed = tokenizer.process(texts[i]) # for some reason when testing interactive python this fails? what is wrong with this process thing
    #print(parsed)
    sents=[line.replace("# text = ","") for line in parsed.split("\n") if line.startswith("# text = ")]
    #print(sents)
    dictionary = ({"id": f"{ids[i]}", "text": f"{sents}"})
    final.append(dictionary)
    if i % 100 == 0:
        save(final)
        final = []

# def save(final):
#     with open("data.json", 'a') as f: # a is for appending
#         for item in final:
#             f.write(json.dumps(item) + "\n")

# OR
with jsonlines.open('output.jsonl', mode='w') as writer:
    for item in final:
        writer.write(item)