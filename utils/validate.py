import os


def valid_file(parser):
    def valid(arg):
        if arg and not os.path.exists(arg):
            parser.error('The file doesn\'t exist: {}'.format(arg))
        else:
            return arg

    return valid
