
# "weights" aka numbers of every bin per label are in the binned_amounts.tsv file
# sampo:  but I was thinking of just applying weights relative
#  to the bin size to the TP, FP, and FN counts for each bin
#   and then summing to take prec, rec, and F.
  

# IF i want everything at once I would have to do a double loop but doable actually?
# cd miscellaneous_stuff/new_batches/sample-01-LABEL
# echo -e "TP \t TN \t FP \t FN" >> ../weighted-results.tsv | for f in * ; do python3 ../../calculate-annotation-results.py ../../pictures_and_other_files/binned_amounts.tsv $f insult 0.5 >> ../weighted-results.tsv; done
# append to file


# this script both counts the TP, FP, TN, FN numbers, weighted or not
#  and if --calculate is used it calculates the precision, recall and f1 from the results file


import sys # is this used?
from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    ap.add_argument('weights', help='binned_amounts.tsv')
    ap.add_argument('annotations', help='tsv')
    ap.add_argument('label', help='e.g. "obscene"')
    ap.add_argument('threshold', help='e.g. 0.5')
    ap.add_argument('--calculate', action="store_true", default=False)
    return ap


# final calculations, read the weighted-results.tsv file and sum up each column and then do the regular prec, rec, f1 calculations
# either use the formulas and do it "manually" or look at if the ready made stuff can use these numbers

# https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea here the micro-avg for precision really helped me understand how to do this
# just sum like I mentioned and do that

def calculate():
    # read file
    with open("unweighted3-results.tsv") as f:
        data = f.readlines()
        data = data[1:]
        for i in range(len(data)):
            data[i] = data[i].replace("\n", "")
            data[i] = data[i].split("\t")

    # loop and sum columns to a new list
    int_data = []
    for i in range(len(data)):
        lista = [eval(i) for i in data[i]]
        int_data.append(lista)
    summed = [sum(i) for i in zip(*int_data)]
    TP = summed[0]
    TN = summed[1]
    FP = summed[2]
    FN = summed[3]

    # these will be MICRO
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)

    print("precision:", precision)
    print("recall:", recall)
    print("f1", f1)

def calculate_labels():
    # read file
    with open("unweighted3-results.tsv") as f:
        data = f.readlines()
        data = data[1:]
        for i in range(len(data)):
            data[i] = data[i].replace("\n", "")
            data[i] = data[i].split("\t")

    # loop and sum columns to a new list
    int_data = []
    for i in range(len(data)):
        lista = [eval(i) for i in data[i]]
        int_data.append(lista)
    
    label_count = 0
    for i in range(0, len(int_data), 10): # steps by ten to get numbers for each label
        summed = [sum(i) for i in zip(*int_data[i:i+10])]
        #print(int_data[i:i+10])
        TP = summed[0]
        TN = summed[1]
        FP = summed[2]
        FN = summed[3]
        #print(TP, TN, FP, FN)

        # these will be MICRO
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * TP / (2 * TP + FP + FN)
        print("label", label_count)
        label_count += 1
        print("precision:", precision)
        print("recall:", recall)
        print("f1", f1)
        print("-----------")


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.calculate:
        calculate()
        calculate_labels()

    else:

        # this has the data from one bin file of the specified label
        # format is [id, label, text, probability]
        with open(args.annotations) as f:
            data = f.readlines()
            data = data[1:]
            for i in range(len(data)):
                data[i] = data[i].replace("\n", "")
                data[i] = data[i].split("\t")
        
        # after this weights has [label, bin1, bin2, etc. ]
        # so when doing this I just need to find the correct row (index) and then loop the rest of the "columns" in that
        with open(args.weights) as f:
            weights = f.readlines()
            weights = weights[1:]
            for i in range(len(weights)):
                weights[i] = weights[i].replace("\n", "")
                weights[i] = weights[i].split("\t")


        # here is where the magic happens lol
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(data)):
            if "not" in data[i][1] and float(data[i][3]) >= float(args.threshold):
                FP += 1
            elif "not" in data[i][1] and float(data[i][3]) < float(args.threshold):
                TN += 1
            elif not "not" in data[i][1] and float(data[i][3]) < float(args.threshold):
                FN += 1
            elif not "not" in data[i][1] and float(data[i][3]) >= float(args.threshold):
                TP += 1

        # get index of the current label that is used
        for i in range(len(weights)):
            if args.label == weights[i][0]:
                index = i
                break

        # multiply the counts with the weights
        for i in range(len(weights[index])):
            if i == 0:
                continue
            else:
                weighted_TP = TP * int(weights[index][i])
                weighted_TN = TN * int(weights[index][i])
                weighted_FP = FP * int(weights[index][i])
                weighted_FN = FN * int(weights[index][i])

        # here print the stuff lol
        #print(TP,"\t", TN, "\t", FP,"\t", FN) # unweighted
        print(weighted_TP, "\t", weighted_TN, "\t", weighted_FP, "\t", weighted_FN)
        # by default the results tsv looks funny because it is split like that due to the bins anyway


if __name__ == '__main__':
    sys.exit(main(sys.argv))