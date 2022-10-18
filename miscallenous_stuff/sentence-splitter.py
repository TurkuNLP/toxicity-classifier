import json
import pandas as pd
import sys
import requests
import re
import datetime

data = sys.argv[1]

# with open(data, 'r') as json_file:
#         json_list = list(json_file)
# lines = [json.loads(jline) for jline in json_list]

df = pd.read_csv(data)
#print(df[:5])

# can I straight read it with read_json(_, orient="records") ????
# df=pd.DataFrame(lines)

#text=False
for i in range(len(df[26700:])):
    ip = str(df["text"][i+26700])
    ip = f"{ip}" # was this necessary? I add quote marks to everything
    #print(ip)

    api_url = f"http://lindat.mff.cuni.cz/services/udpipe/api/process?model=eng&tokenizer&tagger&parser&data={ip}"
    if len(api_url) > 2048:
        print("too long", i + 26700)
        first, second = ip[:int(len(ip)//2)], ip[int(len(ip)//2):]
        api_url1 = f"http://lindat.mff.cuni.cz/services/udpipe/api/process?model=eng&tokenizer&tagger&parser&data={first}"
        response = requests.get(api_url1)
        response = response.json()
        result = response["result"]
        resultlist = result.split("\n")
        api_url2 = f"http://lindat.mff.cuni.cz/services/udpipe/api/process?model=eng&tokenizer&tagger&parser&data={second}"
        response = requests.get(api_url2)
        response = response.json()
        result = response["result"]
        resultlist + result.split("\n") # get the results in one list

    else:
        response = requests.get(api_url)
        response = response.json()
        result = response["result"]
        resultlist = result.split("\n")
    #resultlist = re.split("\n|\t", result)

    textlist = []
    for one in resultlist:
        if "# text " in one:
            #text=True
            newstr = one.replace("# text = ", "") # get just the sentence
        # if "SpacesAfter=" in one:
        #     # could get all spaces from spacesafter thing in the text

            #print("----")
            #print(newstr)   
            textlist.append(newstr)
    
    #print(i+26700)
    # update the dataframe to have the split texts
    df.at[i+26700, 'text'] = textlist 

    num = i + 26700
    # save sometimes just in case
    if num % 1000 == 0:
        now = datetime.datetime.now()
        print(now)
        print(i+26700, "rows split")
        # save to csv
        print(df[26700:num])
        df.to_csv("data/train-sentence-split.csv", index = False)

# save to csv
df.to_csv('data/train-sentence-split.csv', index=False)