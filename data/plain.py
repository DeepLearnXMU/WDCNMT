# plain.py

import numpy

__all__ = ["data_length", "convert_data", "convert_datax", "convert_tag"]


def data_length(line):
    return len(line.strip().split())


def tokenize(data):
    return data.split()


def to_word_id(data, voc, unk="UNK"):
    newdata = []
    unkid = voc[unk]

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    return newdata


def convert_to_array(data, dtype):
    batch = len(data)
    data_len = map(len, data)
    max_len = max(data_len)

    seq = numpy.zeros((max_len, batch), "int32")
    mask = numpy.zeros((max_len, batch), dtype)

    for idx, item in enumerate(data):
        seq[:data_len[idx], idx] = item
        mask[:data_len[idx], idx] = 1.0

    return seq, mask

def convert_datax(data, voc, unk="UNK", eos="<eos>",domaintag="in", dtype="float32"):
    tag = [domaintag for item in data]
    data = [tokenize(item) + [eos] for item in data]
    data = to_word_id(data, voc, unk)

    tag_voc = {"in": 0, "out": 1}

    tdata = to_tag_id(tag, tag_voc)
    seq, mask = convert_to_array(data, dtype)
    tag = convert_tag_to_array(tdata)

    return seq, mask, tag

def to_tag_id(tag, voc):
    newdata = []

    for t in tag:
        newdata.append(voc[t])

    return newdata

def convert_tag_to_array(tag):
    batch = len(tag)
    seq = numpy.zeros((1,batch), "int32")
    for idx, item in enumerate(tag):
        seq[0, idx] = item
    return seq


def convert_data(data, voc, unk="UNK", eos="<eos>", dtype="float32"):
    data = [tokenize(item) + [eos] for item in data]
    data = to_word_id(data, voc, unk)
    seq, mask = convert_to_array(data, dtype)

    return seq, mask

def convert_tag(data, voc):
    data = [tokenize(item)[0] for item in data]
    data = to_tag_id(data, voc)
    tag = convert_tag_to_array(data)
    return tag


