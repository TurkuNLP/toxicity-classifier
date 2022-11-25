import json
import pandas as pd
import sys
import requests
import re
import datetime

thusfar = 48861 # 2 less than the line count is in the csv file

data = sys.argv[1]

df = pd.read_csv(data)
print(df[:5])

# if text begins with & I should remove that character, it messes with the url but does not throw an error
import re
chars_to_remove = ['&', ';', '#']
regular_expression = '[' + re.escape (''. join (chars_to_remove)) + ']'

df["text"] = df['text'].str.replace(regular_expression, '', regex=True)


#text=False
for i in range(len(df[thusfar:])):
    resultlist = []
    ip = str(df["text"][i+thusfar])
    ip = f"{ip}" # was this necessary? I add quote marks to everything
    #print(ip, i)

    api_url = f"http://lindat.mff.cuni.cz/services/udpipe/api/process?model=eng&tokenizer&tagger&parser&data={ip}"
    if len(api_url) > 1000: #2048
        print("too long", i + thusfar)

        # get the strings to lists
        import math
        amount = len(ip) / 1000
        amount = int(math.ceil(amount))
        chunk = int(len(ip) / amount)
        parts = [ip[i:i+chunk] for i in range(0, len(ip), chunk)]
        for new_ip in parts:
            api_url = f"http://lindat.mff.cuni.cz/services/udpipe/api/process?model=eng&tokenizer&tagger&parser&data={new_ip}"
            # print(len(api_url))
            # print(api_url)
            response = requests.get(api_url)
            response = response.json()
            result = response["result"]
            resultlist = resultlist + result.split("\n") # get the results in one list (list + list)
        
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
    df.at[i+thusfar, 'text'] = textlist 

    num = i + thusfar
    # save sometimes just in case
    if num % 10 == 0:
        now = datetime.datetime.now()
        print(now)
        print(i+thusfar, "rows split")
        # save to csv
        print(df[thusfar:num])
        df.to_csv("data/test-sentence-split.csv", index = False)


# save to csv
df.to_csv('data/test-sentence-split.csv', index=False)