#!/usr/bin/env python3

import sys

from collections import Counter, defaultdict
from argparse import ArgumentParser


"""Script for getting the histogram of the probability distributions for the suomi24 data. Not made by me originally."""

def argparser():
    ap = ArgumentParser()
    ap.add_argument('-i', '--interval', type=float, default=0.1)
    ap.add_argument('tsv')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    
    counts = defaultdict(Counter)
    with open(args.tsv) as f:
        header = next(f)
        labels = header.rstrip('\n').split('\t')[1:]    # 1st col is ID
        for line in f:
            values = line.rstrip('\n').split('\t')[1:]
            for label, value in zip(labels, values):
                bin_ = int(float(value)/args.interval)
                counts[label][bin_] += 1

    bins = sorted(set([b for lc in counts.values() for b in lc]))

    itv = args.interval
    bin_range = { b: f'{itv*b:.1f}-{itv*(b+1):.1f}' for b in bins }
    print('\t'.join(['label'] + [bin_range[b] for b in bins]))
    
    for label in counts:
        print('\t'.join([label] + [str(counts[label][b]) for b in bins]))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
