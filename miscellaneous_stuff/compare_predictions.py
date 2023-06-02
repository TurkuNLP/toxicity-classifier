import sys
import pandas as pd

"""See what texts have different labels predicted by different models."""


# python3 miscallenous_stuff/compare_predictions.py predictions/toxic_predicted-multilang.tsv predictions/toxic_predicted-tr.tsv


# get the data
data1 = sys.argv[1]
data2 = sys.argv[2]

#split to lists
def split_data(data):
    with open (data) as f:
        lines = f.readlines()
    line_list = []
    for i in range(len(lines)):
        #lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].split("\t")
        line_list.append(lines[i][0])
    return line_list

first = split_data(data1)
second = split_data(data2)

found = False
same_lines = []
no_same_lines1 = []
for line1 in first:
    for line2 in second:
        if line1 == line2:
            same_lines.append(line1)
            found = True
            break
    if found == False:
        no_same_lines1.append(line1)
    found = False

no_same_lines2 = []
for line2 in second:
    for line1 in first:
        if line1 == line2:
            found = True
            break
    if found == False:
        no_same_lines2.append(line2)
    found = False

dfsame = pd.DataFrame(same_lines, columns=['text'])
dfno1 = pd.DataFrame(no_same_lines1, columns=['text'])
dfno2 = pd.DataFrame(no_same_lines2, columns=['text'])


dfsame.to_csv('comparisons/same_predictions.tsv', sep="\t", header=False, index=False) #, index=False this includes the row number now (to make it clear which example is which) and text)
dfno1.to_csv('comparisons/nosame1_predictions.tsv', sep="\t", header=False, index=False)
dfno2.to_csv('comparisons/nosame2_predictions.tsv', sep="\t", header=False, index=False) 



