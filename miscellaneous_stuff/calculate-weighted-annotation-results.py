
# "weights" aka numbers of every bin per label are in the binned_amounts.tsv file
# sampo:  but I was thinking of just applying weights relative
#  to the bin size to the TP, FP, and FN counts for each bin
#   and then summing to take prec, rec, and F.
  
#   "True positives. This is when a classifier correctly predicts the existence of a label."

#   I have to manually count TP,FP,TN for each bin
#   6 x 10 = 60 bins to count 

#   To make this work almost "automatically" I could do the same bins
#    I have now but also take the predictions
#     and add to the same file -> DONE

#     now I just have to do some calculations
#     make if clause where it checks the annotation and the probability (if probability above or below threshold e.g., 0.5)
#     -> then check whether they match and which it is and assign it to a list (the four possibilities)
#     -> then save the numbers somewhere or read the bin weights and already do some calculations :D 
#     all in all: does not sound too bad? unless I run into some random issues of course.

