# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import theano.tensor as T

from normalize import layer_normalize
from ops import variable_scope, get_variable
from ops.initializer import ones_initializer, zeros_initializer


def embedding_lookup(params, ids):
    shape = list(ids.shape) + [-1]
    values = params[ids.flatten()]
    values = values.reshape(shape)

    return values


# size: [input_size, output_size]
def linear(inputs, size, bias, concat=False, dtype=None, scope=None):
    if not isinstance(size, (list, tuple)):
        raise ValueError("size argument must be (input_size, output_size)")

    input_size, output_size = size

    if not isinstance(input_size, (list, tuple)):
        input_size = [input_size]

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if len(inputs) != len(input_size):
        raise RuntimeError("unmatched elements found: inputs and input_size")

    results = []

    with variable_scope(scope):
        if concat:
            input_size = sum(input_size)
            inputs = theano.tensor.concatenate(inputs, -1)

            shape = [input_size, output_size]
            matrix = get_variable("matrix", shape, dtype=dtype)
            results.append(theano.dot(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = get_variable(name, shape, dtype=dtype)
                results.append(theano.dot(inputs[i], matrix))

        if bias:
            shape = [output_size]
            bias = get_variable("bias", shape, dtype=dtype)
            results.append(bias)

    if len(results) == 1:
        return results[0]

    return reduce(theano.tensor.add, results)


def ln_linear(inputs, size, bias, concat=False, dtype=None, scope=None):
    if not isinstance(size, (list, tuple)):
        raise ValueError("size argument must be (input_size, output_size)")

    input_size, output_size = size

    if not isinstance(input_size, (list, tuple)):
        input_size = [input_size]

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if len(inputs) != len(input_size):
        raise RuntimeError("unmatched elements found: inputs and input_size")

    results = []

    with variable_scope(scope):
        if concat:
            input_size = sum(input_size)
            inputs = theano.tensor.concatenate(inputs, -1)

            shape = [input_size, output_size]
            matrix = get_variable("matrix", shape, dtype=dtype)
            res = theano.dot(inputs, matrix)
            with variable_scope("layer_norm"):
                alpha = get_variable("gains", shape=(output_size,), dtype=dtype, initializer=ones_initializer)
                beta = get_variable("biases", shape=(output_size,), dtype=dtype, initializer=zeros_initializer)

            res = layer_normalize(res, alpha, beta)
            results.append(res)
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = get_variable(name, shape, dtype=dtype)
                res = theano.dot(inputs[i], matrix)
                with variable_scope("layer_norm"):
                    alpha = get_variable("gains_%d" % i, shape=(output_size,), dtype=dtype,
                                         initializer=ones_initializer())
                    beta = get_variable("biases_%d" % i, shape=(output_size,), dtype=dtype,
                                        initializer=zeros_initializer())

                res = layer_normalize(res, alpha, beta)
                results.append(res)

        if bias:
            shape = [output_size]
            bias = get_variable("bias", shape, dtype=dtype)
            results.append(bias)

    if len(results) == 1:
        return results[0]

    return reduce(theano.tensor.add, results)


def feedforward(inputs, size, bias, activation=theano.tensor.nnet.sigmoid,
                concat=False, dtype=None, scope=None, f_linear=linear):
    scope = scope or "feedforward"
    return activation(f_linear(inputs, size, bias, concat, dtype, scope))


def maxout(inputs, size, maxpart, bias, concat=False, dtype=None, scope=None):
    scope = "maxout"
    size[-1] = size[-1] * maxpart

    output = linear(inputs, size, bias, concat, dtype, scope)
    shape = list(output.shape)
    shape[-1] /= maxpart
    shape += [maxpart]
    output = output.reshape(shape)
    output = theano.tensor.max(output, len(shape) - 1)

    return output


def masked_softmax(x, mask=None):
    """
    softmax over axis=1
    there must be at least one 1 in mask
    :param x: 
    :param mask: 
    :return: 
    """
    if x.ndim not in (1, 2) \
            or x.dtype not in T.float_dtypes:
        raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s with ndim=%d' % (x.type, x.ndim))
    if mask and mask.ndim != x.ndim:
        raise ValueError('mask must have the same dim with x. Got x.ndim=%d and mask.ndim=%d' % (x.ndim, mask.ndim))
    if x.ndim == 1:
        x = T.shape_padleft(x, n_ones=1)

    if mask is not None and mask.ndim == 1:
        mask = T.shape_padleft(mask, n_ones=1)

    e_x = T.exp(x - x.max(axis=1)[:, None])
    if mask is not None:
        e_x *= mask

    sm = e_x / e_x.sum(axis=1)[:, None]
    return sm


def masked_softmax2(x, mask=None):
    """
    softmax over axis=1
    deal with case where mask may be all 0
    :param x: 
    :param mask: 
    :return: 
    """
    if x.ndim not in (1, 2) \
            or x.dtype not in T.float_dtypes:
        raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s with ndim=%d' %
                         x.type, x.ndim)
    if mask is not None and mask.ndim != x.ndim:
        raise ValueError('mask must have the same dim with x. Got x=%d-d and mask=%d-d' %
                         x.ndim, mask.ndim)
    if x.ndim == 1:
        x = T.shape_padleft(x, n_ones=1)

    if mask is not None and mask.ndim == 1:
        mask = T.shape_padleft(mask, n_ones=1)

    e_x = T.exp(x - x.max(axis=1)[:, None])
    if mask is not None:
        e_x *= mask
    # avoid 0-division
    denorm = e_x.sum(axis=1) + 1.0 - T.max(mask, axis=1)
    sm = e_x / denorm[:, None]
    return sm
