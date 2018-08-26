#!/usr/bin/python

import argparse
import sys


def main(args):
    unk = args.unk
    for line in sys.stdin:
        if not line:
            break
        words = [w for w in line.split() if w != unk]
        sys.stdout.write(' '.join(words))
        sys.stdout.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default='UNK', help='unk symbol')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
