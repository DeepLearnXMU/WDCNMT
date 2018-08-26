import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict

import pandas


def _join_read(inputs):
    for f in inputs:
        with open(f) as r:
            for l in r:
                yield '%s\n' % l.strip()


def _join_write(inputs, out):
    with open(out, 'w') as w:
        for f in inputs:
            with open(f) as r:
                for l in r:
                    w.write('%s\n' % l.strip())


def join(inputs, out=None):
    if out:
        _join_write(inputs, out)
    else:
        _join_read(inputs)


def infer_ref_stem(src, ref):
    n, remainder = divmod(len(ref), len(src))
    if n < 0 or remainder != 0:
        raise ValueError('mismatched src and refs. (len(src)=%d and len(ref)= %d)' % (len(src), len(ref)))
    ref_stem = []
    for i in xrange(0, len(ref), n):
        ref_i = ref[i:i + n]
        prefix = ref_i[0]
        if n > 1:
            # check trailing number
            try:
                _ = [int(f[-1]) for f in ref_i]
            except ValueError as e:
                raise ValueError('broken format for refs: {}'.format(ref_i))
                # remove trailing number for multiple references case
            prefixes = [f[:-1] for f in ref_i]
            if len(set(prefixes)) > 1:
                raise ValueError('broken format for refs: {}'.format(ref_i))
            prefix = prefixes[0]
        ref_stem.append(prefix)

    return ref_stem


def get_bleu(stdin, script, ref, case):
    p = subprocess.Popen('perl {} {} {}'.format(script, '' if case else '-lc', ref).split(), stdin=stdin,
                         stdout=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        raise RuntimeError()
    bleu = float(out.split('=')[1].split(',')[0].strip())
    return bleu, out


def main(args):
    models = args.models
    entry = args.entry
    inputs = args.inputs
    refs = args.ref
    gpu = args.gpu
    nist = args.nist

    script_translate = args.script_translate
    script_bleu = args.script_bleu

    extra = ''
    if remains:
        extra = ' '.join(remains)

    model_names = [os.path.basename(m) for m in models]

    ref_stems = None

    res1 = []
    for model, model_name in zip(models, model_names):
        print '------------------- {} -------------------'.format(os.path.basename(model))
        # compute signature
        cmd = 'python {script} {model} --sign'.format(script=script_translate, model=model).split()
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode > 0:
            sys.stderr.write(err)
            raise RuntimeError(cmd)
        signature = out.strip()
        # do translate
        cmd = 'python {script} {model} --entry {entry} --inputs {inputs} --gpu {gpu} {extra}'
        cmd = cmd.format(script=script_translate, model=model,
                         entry=entry, inputs=' '.join(inputs),
                         gpu=' '.join(gpu),
                         extra=extra)
        print 'translate script: {}'.format(cmd)
        p = subprocess.Popen(cmd.split())
        p.wait()
        if p.returncode > 0:
            raise RuntimeError(cmd)
        # get output names
        cmd = 'python {script} {model} --entry {entry} --inputs {inputs} --list-outputs'
        cmd = cmd.format(script=script_translate, model=model,
                         entry=entry, inputs=' '.join(inputs))
        p = subprocess.Popen(cmd.split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode > 0:
            sys.stderr.write(err)
            raise RuntimeError(cmd)
        outputs = [fout for fout in out.split(os.linesep) if fout]
        print 'translations:'
        for f in outputs:
            print '\t%s' % f
        # post-process
        if args.post_process_scripts:
            processed = [output + '.post-process' for output in outputs]
            for f1, f2 in zip(outputs, processed):
                temp_out = f2 + '.tmp'
                with open(f1) as r, open(temp_out, 'w') as w:
                    if len(args.post_process_scripts) > 1:
                        p = subprocess.Popen([args.post_process_scripts[0]], stdin=r, stdout=subprocess.PIPE)
                        ps = [p]
                        for script in args.post_process_scripts[1:-1]:
                            print script
                            p_continue = subprocess.Popen([script], stdin=ps[-1].stdout, stdout=subprocess.PIPE)
                            ps.append(p_continue)
                        p_final = subprocess.Popen([args.post_process_scripts[-1]], stdin=ps[-1].stdout, stdout=w)
                        ps.append(p_final)
                        p = ps[-1]
                    else:
                        p = subprocess.Popen([args.post_process_scripts[0]], stdin=r, stdout=w)
                    p.wait()
                    if p.returncode > 0:
                        raise RuntimeError('error occurs in post processing steps.')
                os.rename(temp_out, f2)

            outputs = processed

        if nist:
            exclusion = ['nist02']
            src_34568 = [f for f in outputs if not any(rule in f for rule in exclusion)]
            out_34568 = 'out/translations/{model}-nist-34568-{signature}'.format(model=model_name, signature=signature)
            join(src_34568, out_34568)

            exclusion.append('nist08')
            src_3456 = [f for f in outputs if not any(rule in f for rule in exclusion)]
            out_3456 = 'out/translations/{model}-nist-3456_-{signature}'.format(model=model_name,
                                                                                signature=signature)
            join(src_3456, out_3456)
            outputs+= [out_3456, out_34568]
            outputs = sorted(outputs)
            refs = sorted(refs)

        if refs:
            trans = outputs
            refs = refs
            pattern = re.compile('.*{}-(.*)-{}.*'.format(model_name, signature))
            d1 = OrderedDict()
            d2 = OrderedDict()

            if not ref_stems:
                ref_stems = infer_ref_stem(trans, refs)
            for tran, ref in zip(trans, ref_stems):
                with open(tran) as r:
                    bleu, out = get_bleu(r, script_bleu, ref, args.case)

                name = os.path.basename(tran)
                match = pattern.match(tran)
                if match and match.groups():
                    name = match.groups()[0]
                bleu = float(out.split('=')[1].split(',')[0].strip())
                sys.stdout.write('{}\n\t'.format(name))
                sys.stdout.write(out)
                d1[name] = bleu
            res1.append(d1)

    if args.csv and res1:
        name = args.csv
        if not name.endswith('.csv'):
            name = '%s.csv' % name
        df = pandas.DataFrame(res1, index=model_names)
        df.to_csv(name)
        sys.stderr.write('dumped to %s' % name)


def valid_file(parser, arg):
    arg = os.path.expanduser(arg)
    if arg and not os.path.exists(os.path.expanduser(arg)):
        parser.error('The file doesn\'t exist: {}'.format(arg))
    else:
        return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_type = lambda arg: valid_file(parser, arg)

    parser.add_argument('--models', '-m', nargs='+', type=str)
    parser.add_argument('--entry', '-e', type=file_type, default='rnnsearch.py',
                        help='path to model entry, default to rnnsearch.py')
    parser.add_argument('--inputs', '-i', nargs='+', required=True, type=str)
    parser.add_argument('--ref', '-r', nargs='+', type=str)
    parser.add_argument('--gpu', '-g', nargs='+', default='0', type=str)
    parser.add_argument('--nist', action='store_true')
    parser.add_argument('--post-process-scripts', type=str, nargs='+',
                        help='executable scripts which take as input the stdin')
    parser.add_argument('--csv', type=str)
    parser.add_argument('--case', action='store_true')

    args, remains = parser.parse_known_args()
    args.remains = remains

    scripts_dir = 'scripts'
    args.script_translate = os.path.join(scripts_dir, 'translate.py')
    args.script_bleu = os.path.join(scripts_dir, 'multi-bleu.perl')

    for f in [args.script_translate, args.script_bleu]:
        valid_file(parser, f)

    if args.post_process_scripts:
        for path in args.post_process_scripts:
            if not os.path.exists(path):
                raise ValueError('File doesn\'t exist: %s' % path)
            elif not os.access(path, os.X_OK):
                raise ValueError('File is not executable: %s' % path)

    main(args)
