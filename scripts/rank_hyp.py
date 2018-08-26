import re
import argparse
import multiprocessing

import numpy
import os
import nltk.translate.bleu_score
import itertools
import pandas

import sys


def count_lines(f):
    i = 0
    if not os.path.exists(f):
        return i
    with open(f) as r:
        for _ in r:
            i += 1
    return i


def sentence_bleu((references, hypotheses)):
    scores = []
    for hyp, refs in itertools.izip(hypotheses, references):
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method3
        scores.append(nltk.translate.bleu_score.sentence_bleu(refs, hyp, smoothing_function=smoothing_function))
    return scores


def corpus_bleu((references, hypotheses)):
    return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)


def main(src, hyp, refs, tags, csv):
    n_src = len(src)
    n_ref = len(refs) / len(src)
    n_system = len(hyp) / len(src)

    if not tags:
        tags = range(1, n_system + 1)

    s_lines = []
    for f in src:
        with open(f) as r:
            s_lines.extend(r.readlines())
    h_lines = []
    for i in xrange(n_system):
        h_lines.append([])
        f_hyp = hyp[n_src * i:n_src * (i + 1)]
        for f in f_hyp:
            with open(f) as r:
                for l in r:
                    h_lines[-1].append(l.split())
    r_lines = []
    for i in xrange(n_src):
        fps = [open(f) for f in refs[n_ref * i:n_ref * (i + 1)]]
        for lines in itertools.izip(*fps):
            r_lines.append([l.split() for l in lines])
        for fp in fps:
            fp.close()

    pool1 = multiprocessing.Pool()
    pool2 = multiprocessing.Pool()
    try:
        p1 = pool1.map_async(sentence_bleu, ((r_lines, h_lines[i]) for i in xrange(n_system)))
        p2 = pool2.map_async(corpus_bleu, ((r_lines, _hyp) for _hyp in h_lines))
        scores = p1.get(9999999)
        corpus_scores = p2.get(9999999)

    except KeyboardInterrupt:
        pool1.terminate()
        pool2.terminate()
        pool1.join()
        pool1.close()
        pool2.join()
        pool2.close()
        exit(0)

    scores=numpy.array(scores)

    baselines = scores[:-1]
    target = scores[-1]

    max_baseline_score=numpy.max(baselines,0)
    diff = target-max_baseline_score
    sort_idx=numpy.argsort(diff)[::-1]

    h_lines = zip(*h_lines)
    scores = zip(*scores)

    head_ref = ','.join(['hyp%d' % i for i in xrange(n_ref)])
    head_tag = ','.join(['%s,bleu' % tag for tag in tags])
    header = 'diff,src,{},{}'.format(head_ref, head_tag)

    data = []
    data.append([None] * (2 + n_ref) + list(itertools.chain(*itertools.izip([None] * n_system, corpus_scores))))
    for i in sort_idx:
        i_diff = diff[i]
        i_s = s_lines[i].strip()
        i_refs = [' '.join(words) for words in r_lines[i]]
        i_hyps = [' '.join(words) for words in h_lines[i]]
        i_scores = scores[i]
        data.append([i_diff, i_s] + i_refs + list(itertools.chain(*itertools.izip(i_hyps, i_scores))))

    if not csv:
        csv = 'result.csv'
    df = pandas.DataFrame(data, columns=header.split(','))
    name = csv
    if not name.endswith('.csv'):
        name = '%s.csv' % csv
    df.to_csv(name)
    sys.stderr.write('dumped to %s' % name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, nargs='+', required=True)
    parser.add_argument('--hyp', type=str, nargs='+', required=True)
    parser.add_argument('--refs', type=str, nargs='+', required=True)
    parser.add_argument('--tags', type=str, nargs='+')
    parser.add_argument('--csv', type=str)
    parser.add_argument('--exclude', type=str, action='append')

    args = parser.parse_args()
    if len(args.hyp) % len(args.src) > 0:
        msg = 'number of hypotheses(n=%d) doesn\'t match number of src(n=%d)' % (len(args.hyp), len(args.src))
        raise ValueError(msg)
    if len(args.refs) % len(args.src) > 0:
        msg = 'number of references(n=%d) doesn\'t match number of src(n=%d)' % (len(args.refs), len(args.src))
        raise ValueError(msg)
    if args.tags and len(args.tags) != len(args.hyp) / len(args.src):
        msg = 'number of tags(n=%d) doesn\'t match number of systems(n=%d)' % (
            len(args.tags), len(args.hyp) / len(args.src))
        raise ValueError(msg)

    src = args.src
    hyp = args.hyp
    refs = args.refs
    if args.exclude:
        patterns = [re.compile(rule) for rule in args.exclude]
        src = [f for f in args.src if not any(p.match(f) for p in patterns)]
        hyp = [f for f in args.hyp if not any(p.match(f) for p in patterns)]
        refs = [f for f in args.refs if not any(p.match(f) for p in patterns)]
        exclude = [f for f in args.src if f not in src]
        for f in exclude:
            sys.stderr.write('exclude from src: %s\n' % f)
        exclude = [f for f in args.hyp if f not in hyp]
        for f in exclude:
            sys.stderr.write('exclude from hyp: %s\n' % f)
        exclude = [f for f in args.refs if f not in refs]
        for f in exclude:
            sys.stderr.write('exclude from refs: %s\n' % f)

    main(src, hyp, refs, args.tags, args.csv)
