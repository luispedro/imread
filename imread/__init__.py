# Copyright (C) 2012-2016, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

'''Read and write images as NumPy arrays.

The functions are:

imread
    Read an image from a file.
imread_multi
    Read all the images in a multi-image file (TIFF and STK).
imread_from_blob
    Read an image from a ``bytes`` object in memory.
imsave
    Write an image to a file.
imsave_multi
    Write several images into a single file (TIFF only).
detect_format
    Detect the format of a file or blob from its magic number (PNG and JPEG
    only).
supports_format
    Check whether a format is supported.

``imload``, ``imload_multi``, ``imload_from_blob``, ``imwrite`` and
``imwrite_multi`` are aliases. The ``imread.ijrois`` module reads ImageJ
regions of interest.

Supported formats: PNG, TIFF, JPEG, WebP, BMP, STK, LSM and XCF (read-only,
through ``xcf2png``). Only PNG, TIFF and JPEG can be written.

Documentation: https://imread.readthedocs.io/
'''

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

