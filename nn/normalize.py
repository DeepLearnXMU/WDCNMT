import theano.tensor as T


def layer_normalize(x, alpha, beta):
    """
    layer normalization
    :param x: 2d tensor of shape (batch, dim)
    :param alpha: gains vector of shape (dim,)
    :param beta: biases vector of shape (dim,)
    :return:
    """
    assert x.ndim == 2
    _eps = 1e-6
    mean = T.mean(x, 1)
    var = T.var(x, 1)
    std = T.sqrt(var[:, None] + _eps)
    output = (x - mean[:, None]) / std
    output = alpha[None, :] * output + beta[None, :]
    return output
