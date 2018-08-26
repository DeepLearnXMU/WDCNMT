# random.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano.sandbox.rng_mrg

_RANDOM_STREAM = theano.sandbox.rng_mrg.MRG_RandomStreams()

import itertools


def seed(seed):
    _RANDOM_STREAM.seed(seed)


def get_state():
    return trng_get_state(_RANDOM_STREAM)


def set_state(state):
    trng_set_state(state, _RANDOM_STREAM)


def trng_get_state(trng):
    return [tup[0].get_value() for tup in trng.state_updates]


def trng_set_state(state, trng, suppress=True):
    if len(state) == len(trng.state_updates):
        for s, tup in itertools.izip(state, trng.state_updates):
            tup[0].set_value(s)
    else:
        if not suppress:
            raise ValueError(
                "Expected state(n_var=%d) doesn't fit trng(n_var=%d)" % (len(state), len(trng.state_updates)))


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    return _RANDOM_STREAM.normal(shape, mean, stddev, dtype=dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    return _RANDOM_STREAM.uniform(shape, minval, maxval, dtype=dtype)


def binomial(shape, prob, num_samples=1, dtype=None, seed=None):
    return _RANDOM_STREAM.binomial(shape, num_samples, prob, dtype=dtype)


def multinomial(dist, num_samples=1, seed=None):
    if dist.ndim != 2:
        raise ValueError("dist is assumed to have shape [batch, dim]")

    return _RANDOM_STREAM.multinomial(n=num_samples, pvals=dist)
