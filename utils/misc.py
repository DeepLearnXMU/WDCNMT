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
                raise ValueError("broken format for refs: {}".format(ref_i))
                # remove trailing number for multiple references case
            prefixes = [f[:-1] for f in ref_i]
            if len(set(prefixes)) > 1:
                raise ValueError("broken format for refs: {}".format(ref_i))
            prefix = prefixes[0]
        ref_stem.append(prefix)

    return ref_stem
