import argparse
from collections import defaultdict
import cPickle
import hashlib

import itertools

import sys


def main(args):
    corpus = args.corpus
    f_vocabs = args.vocabs
    unk = args.unk
    ratio = args.ratio
    dedup = args.dedup

    if f_vocabs:
        vocabs = [cPickle.load(open(f, 'rb')) for f in f_vocabs]
    else:
        vocabs = [None] * len(corpus)

    assert len(corpus) == len(vocabs)

    fds = [open(f) for f in corpus]

    hashes = [set() for _ in corpus]

    writers = [open('%s.filtered' % f, 'w') for f in corpus]

    record_dup = defaultdict(int)
    record_ratio = 0
    while True:
        try:
            lines = [fd.next() for fd in fds]
        except StopIteration:
            break

        newlines = []
        delete = False

        isdup = [False] * len(lines)
        for i, line in enumerate(lines):
            vocab = vocabs[i]
            if vocab:
                words = line.split()
                words2 = [w if w in vocabs[i] else unk for w in words]
                n_l = len(words)
                ratio_l = words2.count(unk) / (n_l + 0.0)

                if ratio_l >= ratio:
                    delete = True
                    record_ratio += 1
                    break

                newlines.append(' '.join(words2))
            else:
                newlines.append(line.strip())

            if dedup:
                hash_val = hashlib.md5(newlines[-1]).hexdigest()
                isdup[i] = hash_val in hashes[i]
                hashes[i].add(hash_val)

        if dedup and all(isdup):
            delete = True
        if not delete:
            for writer, line in itertools.izip(writers, newlines):
                writer.write('%s\n' % line)

    for fd in fds + writers:
        fd.close()

    print 'delete ratio >= %.2f: %d' % (ratio, record_ratio)
    print 'delete dups: %d' % len(record_dup)
    for k, v in record_dup.iteritems():
        sys.stderr.write('%d\t%s\n' % (v, k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str, nargs='+')
    parser.add_argument('--vocabs', type=str, nargs='+')
    parser.add_argument('--unk', type=str, default='UNK')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='discard sentence pairs with unk ratio greater or equal than given value')
    parser.add_argument('--dedup', action='store_true', help='remove duplicates')
    parser.add_argument('--replace', action='store_true', help='replace oov word with unk symbol')

    args = parser.parse_args()
    main(args)

