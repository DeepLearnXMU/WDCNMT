import itertools

import numpy


def _join_read(inputs):
    for f in inputs:
        with open(f) as r:
            for l in r:
                yield "%s\n" % l.strip()


def _join_write(inputs, out):
    with open(out, 'w') as w:
        for f in inputs:
            with open(f) as r:
                for l in r:
                    w.write("%s\n" % l.strip())


def join(inputs, out=None):
    if out:
        _join_write(inputs, out)
    else:
        _join_read(inputs)


def shuffle(inputs, outputs, indices=None, strict=False):
    """
    shuffle with low memory usage
    :param inputs: 
    :param outputs: 
    :param indices: 
    :param strict: 
    :return: 
    """
    if isinstance(inputs, (str)):
        inputs = [inputs]
    if isinstance(outputs, (str)):
        outputs = [outputs]

    if len(inputs) != len(outputs):
        raise ValueError("The number of given inputs and outputs doesn't match.")
    counts = []
    offsets = []
    for f in inputs:
        c = 0
        offs = []
        r = open(f, 'r')
        off = 0
        r.readline()
        while r.tell() > off:
            off = r.tell()
            offs.append(off)
            c += 1
            r.readline()
        if c > 0:
            offs = [0] + offs[:-1]

        counts.append(c)
        offsets.append(offs)
    if strict:
        if len(set(counts)) != 1:
            raise ValueError("The files are not strictly aligned.")
    else:
        count = min(counts)
        counts = [count] * len(inputs)
        offsets = [offs[:count] for offs in offsets]

    offsets = numpy.array(offsets)
    if indices is None:
        indices = numpy.arange(counts[0])
        numpy.random.shuffle(indices)

    indices = numpy.array(indices)

    if indices.size != counts[0]:
        raise ValueError("The given indices doesn't match lines of file.")

    offsets = offsets[:, indices]  # shuffled offsets
    for i, fi, fo, in itertools.izip(itertools.count(), inputs, outputs):
        with open(fi) as r, open(fo, 'w') as w:
            for off in offsets[i]:
                r.seek(off)
                line = r.readline()
                w.write("%s\n" % line.strip())
    return indices
