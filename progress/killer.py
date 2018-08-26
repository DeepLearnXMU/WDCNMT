import signal


# https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully

def register_killer():
    def raise_exception(signum, frame):
        raise KilledException('Process receives a signal: %d.' % signum)

    signal.signal(signal.SIGINT, raise_exception)
    signal.signal(signal.SIGTERM, raise_exception)


class GracefulKiller:
    def __init__(self):
        signal.signal(signal.SIGINT, self.raise_exception)
        signal.signal(signal.SIGTERM, self.raise_exception)

    def raise_exception(self, signum, frame):
        raise KilledException('Process receives a termination signal.')


class KilledException(Exception):
    pass
