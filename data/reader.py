# reader.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import tempfile

import numpy

from utils.file import shuffle


# lowest-level stream
class textreader:
    def __init__(self, name, shuffle=False, strict=True):
        if not isinstance(name, (list, tuple)):
            name = [name]

        self.eos = False
        self.count = 0
        self.names = name
        self.stream = []
        self._shuffle = shuffle
        self.strict = strict
        self.tempfiles = None
        self.indices = None

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def readline(self):
        # read from file
        texts = [fd.readline() for fd in self.stream]
        flag = any([line == "" for line in texts])

        if flag:
            return None

        self.count += 1
        texts = [text.strip() for text in texts]

        return texts

    def next(self):
        data = self.readline()

        if data is None:
            self.reset()
            raise StopIteration

        return data

    def shuf(self, indices=None):
        """
        dump shuffled files to temp files and return sorting indices
        """
        # read original file
        stream = [open(item, "r") for item in self.names]
        texts = [fd.readlines() for fd in stream]

        if indices is None:
            linecnt = min([len(text) for text in texts])
            indices = numpy.arange(linecnt)
            numpy.random.shuffle(indices)
        shuffle(self.names, [name for _, name in self.tempfiles], indices, self.strict)

        return indices

    def reset(self):
        self.count = 0
        self.eos = False

        if not self.stream:
            self.stream = [open(item, "r") for item in self.names]

        for fd in self.stream:
            fd.seek(0)

        if self._shuffle:
            self.close()
            self.tempfiles = [tempfile.mkstemp() for _ in self.names]

            self.indices = self.shuf()
            self.stream = [open(name, "r") for (_, name) in self.tempfiles]

    def close(self):
        for fd in self.stream:
            fd.close()
        if self._shuffle and self.tempfiles:
            for fd, name in self.tempfiles:
                # on Windows, it's required to release both file object and descriptor
                os.close(fd)
                os.remove(name)
            self.tempfiles = None
            self.stream = []

    def get_indices(self):
        return self.indices

    def set_indices(self, indices):
        self.indices = indices

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["stream"]
        del d["tempfiles"]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._shuffle:
            self.tempfiles = [tempfile.mkstemp() for _ in self.names]
            self.shuf(self.indices)
            self.stream = [open(name, "r") for (_, name) in self.tempfiles]
        else:
            self.stream = [open(name, "r") for name in self.names]
        for _ in xrange(self.count):
            for stream in self.stream:
                stream.readline()
