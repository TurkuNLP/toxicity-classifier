#!/usr/bin/env python3

# mkdir sample-01-obscene
# for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do h=$(python -c 'print('$l'+0.1)'); python3 sample_comments.py suomi24-2001-2020-1p-sample.jsonl s24predictions.tsv obscene $l $h 10 > sample-01-obscene/comments_${l}-${h}.jsonl; done

#for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do h=$(python3 -c 'print('$l'+0.1)'); python3 sample_comments.py ../annotations/all_annotations.tsv ../predictions/s24predictions.tsv insult $l $h > annotation-batches/sample-01-insult/comments_${l}-${h}.tsv; done

import sys
import json

from logging import error
from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    ap.add_argument('comments', help='jsonl')
    ap.add_argument('predictions', help='tsv')
    ap.add_argument('label', help='e.g. "obscene"')
    ap.add_argument('low', type=float, help='e.g. 0.0')
    ap.add_argument('high', type=float, help='e.g. 0.1')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    ids = []
    with open(args.predictions) as f:
        header = next(f).rstrip('\n')
        try:
            index = header.split('\t').index(args.label)
        except ValueError:
            error(f'label "{args.label}" not found in header "{header}"')
            return 1

        for ln, line in enumerate(f, start=2):
            fields = line.rstrip('\n').split('\t')
            id_, value = fields[0], fields[index]
            value = float(value)
            if args.low <= value <= args.high:
                ids.append(id_)
    
    with open(args.comments, "rt", encoding="utf-8") as f:
        data = f.readlines()
    data = data[1:]
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        data[i] = data[i].split("\t")
        if data[i][0] in ids:
            if args.label in data[i][1]: #"toxicity" == data[i][1] or "not-toxicity" == data[i][1]:
                print(data[i], end='\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
