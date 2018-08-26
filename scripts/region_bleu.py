import argparse
import itertools
import math
import re
import os
import subprocess
import sys
import uuid
from collections import OrderedDict, defaultdict


def count_lines(f):
    i = 0
    if not os.path.exists(f):
        return i
    with open(f) as r:
        for _ in r:
            i += 1
    return i


def main(args):
    srcs = args.src
    trans = args.trans
    refs = args.ref
    nref = args.nref
    split = args.split
    chunk = args.chunk
    script = args.script
    method = args.method

    src_fds = [open(src) for src in srcs]
    src_it = itertools.chain(*src_fds)
    trans_fds = [open(t) for t in trans]
    trans_it = itertools.chain(*trans_fds)
    ref_fds = [[open(f) for f in itertools.islice(refs, i, len(refs), nref)]
               for i in xrange(nref)]
    ref_it = itertools.izip(*[itertools.chain(*fds) for fds in ref_fds])

    # group by length
    len2trans = defaultdict(list)
    len2refs = defaultdict(list)
    for x, y, ys in itertools.izip(src_it, trans_it, ref_it):
        length = len(x.split())
        len2trans[length].append(y.strip())
        len2refs[length].append([y_.strip() for y_ in ys])

    for fd in itertools.chain(src_fds, trans_fds, *ref_fds):
        fd.close()

    n0 = sum(map(len, len2refs.itervalues()))
    n1 = sum(map(len, len2trans.itervalues()))
    if n0 != n1:
        raise RuntimeError("Lines mismatch.(n-ref={}, n-trans={})".format(n0, n1))

    maxlen = max(len2trans.iterkeys())

    # setup interval
    if split:
        interval = split
    elif chunk is not None and chunk > 0:
        n = int(math.ceil(maxlen / (chunk + 0.0)))
        interval = [chunk] * n
        interval = [sum(interval[:i + 1]) for i, _ in enumerate(interval)]
    else:
        raise RuntimeError()

    if interval[0] > 0:
        interval = [0] + interval
    if interval[-1] < maxlen:
        interval = interval + [maxlen]
    else:
        interval[-1] = maxlen

    if len(interval) > 3 and (maxlen - interval[-2] + 0.0) / (interval[-2] - interval[-3]) < 0.5:
        interval = interval[:-3] + [interval[-1]]

    int2trans = OrderedDict()
    sorted_len2trans = [(length, len2trans[length]) for length in sorted(len2trans.iterkeys())]
    sorted_len2refs = [(length, len2refs[length]) for length in sorted(len2trans.iterkeys())]

    int2refs = OrderedDict()
    for lower, upper in itertools.izip(interval[:-1], interval[1:]):
        lines = []
        refs = []
        for (length, i_lines), (_, i_refs) in itertools.izip(sorted_len2trans,
                                                             sorted_len2refs):
            if method == 'less':
                if length <= upper:
                    lines.extend(i_lines)
                    refs.extend(i_refs)
                else:
                    break
            elif method == 'greater':
                if length >= lower:
                    lines.extend(i_lines)
                    refs.extend(i_refs)
            elif method == 'between':
                if length > lower and length <= upper:
                    lines.extend(i_lines)
                    refs.extend(i_refs)

        int2trans[lower, upper] = lines
        int2refs[lower, upper] = refs

    n = 0
    tmpname = uuid.uuid4()
    tmp_refs = ['%s%d' % (tmpname, i) for i in xrange(nref)]
    for (lower, upper), lines in int2trans.iteritems():
        n_diff = len(lines) - n
        n = len(lines)
        sys.stdout.write('{}-{}[n={}, {:+}] '.format(lower, upper, n, n_diff))

        if len(lines) == 0:
            print ""
            continue

        try:
            fds = [open(tmp_ref, 'w') for tmp_ref in tmp_refs]
            for item in int2refs[lower, upper]:
                for fd, y_ref in itertools.izip(fds, item):
                    fd.write('%s\n' % y_ref)
            for fd in fds:
                fd.close()

            p = subprocess.Popen('perl {} -lc {}'.format(script, tmpname).split(),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate('\n'.join(lines))
        finally:
            for f in tmp_refs:
                os.remove(f)

        if p.returncode > 0:
            print err.strip()
        else:
            print output.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', nargs='+', required=True)
    parser.add_argument('--ref', nargs='+', required=True)
    parser.add_argument('--nref', type=int, required=True)
    parser.add_argument('--trans', nargs='+', required=True)
    parser.add_argument('--script', default='scripts/multi-bleu.perl', help='path to bleu script')
    parser.add_argument('--split', nargs='+', type=int, help='less or equal to')
    parser.add_argument('--chunk', type=int)
    parser.add_argument('--exclude', type=str)
    parser.add_argument('--method', choices=['between', 'greater', 'less'],
                        default='between',
                        help='between for between interval, greater for greater than lower bound, less for less than upper bound')
    args = parser.parse_args()

    # parse args
    if args.exclude:
        args.src=[f for f in args.src if not re.match(args.exclude,f)]
        args.ref=[f for f in args.ref if not re.match(args.exclude,f)]
        args.trans=[f for f in args.trans if not re.match(args.exclude,f)]


    src = args.src
    trans = args.trans


    if not args.split and not args.chunk:
        raise argparse.ArgumentError(args.split, "One of arg (split, chunk) needs to be provided")

    n_src = sum(map(count_lines, src))
    n_trans = sum(map(count_lines, trans))
    if n_src != n_trans:
        raise argparse.ArgumentError(args.trans, "Lines mismatch.(n-src={}, n-trans={})".format(n_src, n_trans))

    return args


if __name__ == '__main__':
    args = parse_args()
    print 'method = %s\n' % args.method
    main(args)
