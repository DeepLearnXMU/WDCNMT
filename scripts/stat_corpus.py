import argparse
import cPickle
import itertools
import os

from collections import defaultdict

import math


def main(args):
    corpus = args.corpus
    vocabs = args.vocabs
    split = args.split
    chunk = args.chunk

    if vocabs:
        vocabs = [cPickle.load(f) for f in vocabs]
    else:
        vocabs = [None] * len(corpus)

    for f, vocab in itertools.izip(corpus, vocabs):
        int2tok = defaultdict(int)
        int2invoc = defaultdict(int)
        int2line = defaultdict(int)
        l_min = 66666
        l_max = 0
        n_line = 0.0
        with open(f) as r:
            for line in r:
                words = line.split()
                length = len(words)
                l_min = min(length, l_min)
                l_max = max(length, l_max)
                n_line += 1
        if split:
            interval = split
        elif chunk is not None and chunk > 0:
            n = int(math.ceil(l_max / (chunk + 0.0)))
            interval = [chunk] * n
            interval = [sum(interval[:i + 1]) for i, _ in enumerate(interval)]
        else:
            raise RuntimeError()

        if interval[0] > 0:
            interval = [0] + interval
        if interval[-1] < l_max:
            interval = interval + [l_max]
        else:
            interval[-1] = l_max

        if len(interval) > 3 and (l_max - interval[-2] + 0.0) / (interval[-2] - interval[-3]) < 0.5:
            interval = interval[:-3] + [interval[-1]]
        with open(f) as r:
            for line in r:
                words = line.split()
                length = len(words)
                for (lower, upper) in itertools.izip(interval[:-1], interval[1:]):
                    if length > lower and length <= upper:
                        int2tok[lower, upper] += length
                        int2line[lower, upper] += 1
                        if vocab:
                            n_in_voc = len([word for word in words if word in vocab])
                            int2invoc[lower, upper] += n_in_voc

        print '------- %s -------' % os.path.basename(f)
        print 'lines:{:,}, tokens:{:,}, min length: {}, max length: {}'.format(int(n_line), sum(int2tok.values()), l_min,
                                                                               l_max)

        for (lower, upper) in itertools.izip(interval[:-1], interval[1:]):
            p_line = int2line[lower, upper] / n_line
            p_tok = (int2tok[lower, upper] + 0.0) / sum(int2tok.values())
            if vocab:
                coverage = (int2tok[lower, upper] + 0.0) / int2invoc[lower, upper]
                print '({}, {}]: lines: {:.2%}, tokens: {:.2%}, coverage: {:.2%}'.format(lower, upper, p_line, p_tok,
                                                                                         coverage)

            else:
                print '({}, {}]: lines: {:.2%}, tokens: {:.2%}'.format(lower, upper, p_line, p_tok)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str, nargs='+')
    parser.add_argument('--vocabs', type=str, nargs='+')
    parser.add_argument('--split', type=int, nargs='+')
    parser.add_argument('--chunk', type=int, default=10)
    args = parser.parse_args()
    assert args.vocabs is None or len(args.corpus) == len(args.vocabs)
    main(args)

