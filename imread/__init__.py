# Copyright (C) 2012-2016, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

try:
    from .imread import imread, imwrite, imread_multi, imread_from_blob, imwrite_multi
    from .imread import detect_format, supports_format
    from .imread import imload, imsave, imload_multi, imload_from_blob, imsave_multi
    from .imread_version import __version__
except ImportError as e:
    import sys
    print('''\
Could not import submodules (exact error was: {0}).

The most common cause is importing imread from inside its source directory,
which picks up the source tree (without the compiled extension) instead of the
installed package.

Try installing imread (see https://imread.readthedocs.io/) and then changing to
another directory before importing it. For development, use an editable
install (`make debug`).
'''.format(e), file=sys.stderr)

