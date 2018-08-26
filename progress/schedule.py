import threading

from functools import wraps


def delay(delay=0.):
    """
    Decorator delaying the execution of a function for a while.
    example:
        def get_repeater(time_delay):
            @delay(time_delay)
            def repeat(*args, **kwargs):
                start(*args, **kwargs)

    """

    def wrap(f):
        @wraps(f)
        def delayed(*args, **kwargs):
            timer = threading.Timer(delay, f, args=args, kwargs=kwargs)
            timer.start()

        return delayed

    return wrap


def setTimeout(callback, sleep):
    @delay(sleep)
    def f():
        callback()

    f()
