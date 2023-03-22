import json
import jsonlines
import sys
import datasets

#srun python3 miscallenous_stuff/sentence-splitter.py data/train_en.jsonl #data/test_en.jsonl

data = sys.argv[1]

dataset = datasets.load_dataset("json", data_files=data)
print(dataset)

texts = dataset["train"]['text'] # automatically loads as train split in the above
ids = dataset["train"]['id']
final = []

thusfar = 0 # the line number where to continue if the pipe fails or something


# def save(final):
#     with open("data.json", 'a') as f: # a is for appending
#         for item in final:
#             f.write(json.dumps(item) + "\n")

# OR
def save(final):
    with jsonlines.open(data, mode='a') as writer:
        for item in final:
            writer.write(item)


import ufal.udpipe
udpipemodel=ufal.udpipe.Model.load("miscallenous_stuff/udpipe/english-ewt.udpipe") # change to a finnish model for backtranslation
tokenizer=ufal.udpipe.Pipeline(udpipemodel,"tokenizer=ranges","none","none","conllu")
for i in range(len(texts)):
    parsed = tokenizer.process(texts[i])
    #print(parsed)
    sents=[line.replace("# text = ","") for line in parsed.split("\n") if line.startswith("# text = ")]
    #print(sents)
    dictionary = ({"id": f"{ids[i]}", "text": f"{sents}"})
    print(dictionary)
    final.append(dictionary)
    if i % 1000 == 0 and i != 0:
        print(f"{i} saved")
        save(final)
        final = []