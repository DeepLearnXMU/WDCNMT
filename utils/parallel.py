import multiprocessing

from utils.type import is_lambda


def _worker(delegate, queue, rqueue):
    while True:
        i, datum = queue.get()
        if i < 0:
            break
        rv = delegate(datum)
        rqueue.put((i, rv))


def parallel(data, delegate, spawn=2):
    if not hasattr(data, "__iter__"):
        raise ValueError("Type %s is not iterable" % type(data))
    if spawn < 2:
        raise ValueError("Illegal spawn: %d" % spawn)
    if is_lambda(delegate):
        raise ValueError("Illegal type for delegate: lambda expression")

    queue, rqueue = multiprocessing.Queue(), multiprocessing.Queue()
    # worker=self.wrap_worker(self.proc)
    ps = [multiprocessing.Process(target=_worker, args=(delegate, queue, rqueue))]
    for p in ps:
        p.daemon = True
        p.start()
    n = 0
    for i, datum in enumerate(data):
        queue.put((i, datum))
        n += 1
    j = 0
    buffer = []
    while j < n:
        if any([p.exitcode > 0 for p in ps]):  # error in subprocess
            raise RuntimeError()
        added = False
        while rqueue.qsize() > 0:
            buffer.append(rqueue.get())
            added = True
        if added:
            buffer = sorted(buffer, key=lambda x: x[0])
        while buffer and buffer[0][0] == j:
            i, datum = buffer[0]
            yield datum
            j += 1
            buffer = buffer[1:]
