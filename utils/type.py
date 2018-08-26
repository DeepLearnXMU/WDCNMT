from collections import OrderedDict, defaultdict, deque

import numpy

_PRIMITIVES_SET_1 = (int, float, str, bool, numpy.ndarray)  # primitive type
_PRIMITIVES_SET_2 = (list, tuple, set, deque)  # iterable type
_PRIMITIVES_SET_3 = (dict, OrderedDict, defaultdict)  # key-value pair type


def is_pickable(var, recursive=False):
    if isinstance(var, _PRIMITIVES_SET_1):
        return True
    if not recursive:
        return isinstance(var, _PRIMITIVES_SET_2 + _PRIMITIVES_SET_3)
    else:
        if isinstance(var, _PRIMITIVES_SET_2):
            for item in var:
                if not is_pickable(item, recursive):
                    return False
        elif isinstance(var, _PRIMITIVES_SET_3):
            for k, v in var.iteritems():
                if not is_pickable(k, recursive) or not is_pickable(v, recursive):
                    return False
        else:
            return False
        return True


def is_lambda(func):
    LAMBDA = lambda: 0
    return isinstance(func, type(LAMBDA)) and func.__name__ == LAMBDA.__name__
