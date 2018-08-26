import argparse
import hashlib
import itertools
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

from collections import OrderedDict, deque


def count_lines(f):
    i = 0
    if not os.path.exists(f):
        return i
    with open(f) as r:
        for _ in r:
            i += 1
    return i


def worker(queue, rqueue, entry, model, device, extra, quiet):
    env = dict(os.environ)  # Make a copy of the current environment

    if device != "cpu" and not device.startswith("gpu"):
        device = "gpu%s" % device

    if isinstance(extra, (list, tuple)):
        extra = " ".join(extra)

    env['THEANO_FLAGS'] = 'device={}'.format(device)
    # import theano.sandbox.cuda
    # theano.sandbox.cuda.use(gpu)

    cmd = 'python -u {entry} translate --model {model} {extra}'.format(
        entry=entry,
        model=model,
        extra=extra)
    if not quiet:
        sys.stderr.write("translate cmd: {}\n".format(cmd))

    p = subprocess.Popen(cmd.split(),
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=open(os.devnull, 'w'),
                         env=env)
    while True:
        i, line = queue.get()
        if i < 0:  # end of queue
            break
        p.stdin.write("%s\n" % line.strip())
        out = p.stdout.readline()
        rqueue.put((i, out))


def translate(model, signature, pending, done, src2out, devices, entry, extra, quiet, write_stdout):
    n_pending = map(count_lines, pending)
    n_done = sum(map(count_lines, done)) if done else 0
    n_total = sum(n_pending) + n_done

    if sum(n_pending) == 0:
        return

    out_dir = os.path.dirname(src2out.values()[0])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if pending:
        tic = time.time()

        fds = [open(f) for f in pending]
        reader = itertools.chain(*fds)

        queue = multiprocessing.Queue()
        rqueue = multiprocessing.Queue()

        ps = [multiprocessing.Process(target=worker, args=(queue, rqueue, entry, model, device, extra, quiet)) for
              device in
              devices]
        try:
            for p in ps:
                p.daemon = True
                p.start()
            # feed
            for i, line in enumerate(reader):
                queue.put((i, line))

            for _ in ps:
                queue.put((-1, None))  # end of queue

            # printing
            hist = deque(maxlen=5)  # average over 5 past records
            # consume to prevent holding all translations in memory
            buffer = []
            i = 0
            j = 0
            writer = None
            while i < len(pending):
                time.sleep(1)
                # check exitcode and only get() when queue.qsize()>0 to prevent dead lock when error encountered in subprocess
                if any([p.exitcode > 0 for p in ps]):  # error in subprocess
                    sys.stderr.write("Error occurs in worker")
                    raise RuntimeError()

                added = False
                while rqueue.qsize() > 0:
                    buffer.append(rqueue.get())
                    added = True

                if added:
                    buffer = sorted(buffer, key=lambda x: x[0])

                while buffer and buffer[0][0] == sum(n_pending[:i]) + j:
                    idx, trans = buffer[0]
                    if writer is None:
                        if write_stdout:
                            writer = sys.stdout
                        else:
                            writer = open(src2out[pending[i]], 'w')
                    writer.write(trans)
                    j += 1
                    if not j < n_pending[i]:
                        if not write_stdout and writer is not None:
                            writer.close()
                            writer = None
                        i += 1
                        j = 0
                    buffer = buffer[1:]  # remove processed output

                if not quiet:
                    n1 = n_done + sum(n_pending[:i]) + j + rqueue.qsize() + len(buffer)
                    hist.append(n1)
                    rate = 0.0
                    if len(hist) > 1:
                        rate = (hist[-1] - hist[0] + 0.0) / len(hist)
                    toc = time.time()
                    sys.stderr.write(
                        '\r{}/{}, {:.2f} s/sec, {}'.format(n1, n_total, rate,
                                                           time.strftime('%H:%M:%S', time.gmtime(toc - tic))))

            if not quiet:
                sys.stderr.write('\n')
        except KeyboardInterrupt:
            traceback.print_exc()
            for p in ps:
                p.terminate()
                p.join()
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()
            for p in ps:
                p.terminate()
                p.join()
            sys.exit(1)


def main(args):
    model = args.model
    entry = args.entry
    inputs = args.inputs
    tag = args.tag
    gpu = args.gpu
    quiet = args.quiet
    remains = args.remains
    write_stdout = args.stdout
    force = args.force

    extra = " ".join(remains)

    if gpu:
        sp = [item.split(':') for item in gpu]
        sp = [item if len(item) == 2 else [item[0], '1'] for item in sp]
        devices = list(itertools.chain(*[itertools.repeat(id, int(n_task)) for id, n_task in sp]))
    else:
        devices = ['cpu']

    signature = hashlib.md5(open(model, 'rb').read()).hexdigest()
    if tag:
        signature = '%s-%s' % (signature, tag)

    if args.signature:
        print signature
        sys.exit(0)

    src_signature = [hashlib.md5(open(f, 'rb').read()).hexdigest() for f in inputs]

    # skip translated
    src2out = OrderedDict()
    for s, s_sign in itertools.izip(inputs, src_signature):
        output = os.path.join('out', 'translations',
                              '{}-{}-{}-{}'.format(os.path.basename(model), os.path.basename(s), signature, s_sign))
        src2out[s] = output

    if args.list_outputs:
        for output in src2out.itervalues():
            print output
        sys.exit(0)

    pending = []
    done = []
    if force:
        pending = inputs
    else:
        for s, o in src2out.iteritems():
            if os.path.exists(o) and count_lines(s) == count_lines(o):
                # skip translated
                done.append(s)
            else:
                pending.append(s)

    if not quiet:
        for f in done:
            sys.stderr.write('skip {}\n'.format(f))

    translate(model, signature, pending, done, src2out, devices, entry, extra, quiet, write_stdout)


def valid_file(parser, arg):
    if arg and not os.path.exists(arg):
        parser.error('The file doesn\'t exist: {}'.format(arg))
    else:
        return arg


def parse_args():
    parser = argparse.ArgumentParser()
    file_type = lambda arg: valid_file(parser, arg)

    parser.add_argument('model')
    parser.add_argument('--entry', '-e', default='rnnsearch.py')
    parser.add_argument('--inputs', '-i', type=file_type, nargs='+')
    parser.add_argument('--sign', action='store_true', dest='signature', help='print signature and exit')
    parser.add_argument('--list-outputs', action='store_true',
                        help='list output names in correspondence with given model and input files, then exit')
    parser.add_argument('--tag', '-t', type=str)
    parser.add_argument('--gpu', '-g', nargs='+', type=str, help='e.g. --gpu 0:3 1:2 or --gpu 0 1')
    parser.add_argument('--quiet', '-q', action="store_true", help='suppress procedural printings')
    parser.add_argument('--stdout', action="store_true",
                        help='write to stdout instead of files, if True, suppress all irrelevant stdout')
    parser.add_argument('--force', '-f', action="store_true", help='force to translate all input files')

    args, remains = parser.parse_known_args()
    args.remains = remains

    if not args.signature:
        valid_file(parser, args.entry)
        if not args.inputs:
            parser.error('argument --src is required')

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
