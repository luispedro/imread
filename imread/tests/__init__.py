def file_path(fname):
    from os import path
    return path.join(
            path.dirname(__file__),
            'data',
            fname)
