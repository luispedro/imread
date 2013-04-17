# -*- coding: utf-8 -*-
# Copyright (C) 2012-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

import numpy as np

from . import _imread

from .special import special

def _parse_formatstr(filename, formatstr, funcname):
    if formatstr is not None:
        return formatstr
    from os import path
    _,ext = path.splitext(filename)
    if len(ext) and ext[0] == '.':
        return ext[1:].lower()
    raise ValueError('imread.%s: Could not identify format from filename: `%s`' % (funcname,filename))

def imread(filename, as_grey=False, formatstr=None):
    '''
    im = imread(filename, as_grey=False, formatstr={filename extension})

    Read an image into a ndarray.

    Parameters
    ----------
    filename : str
        filename
    as_grey : boolean, optional
        Whether to convert to grey scale image (default: no)
    formatstr : str, optional
        Format name. This is typically the same as the extension of the file
        and inferred from there. However, if you have a file whose extension
        does not correspond to the format, you can pass it explicitly.

    Returns
    -------
    im : ndarray
        The type of this array will depend on the contents of the file and of
        `as_grey`. Conversion from colour to grayscale will return a floating
        point image.
    '''
    formatstr = _parse_formatstr(filename, formatstr, 'imread')
    reader = special.get(formatstr, _imread.imread)
    im = reader(filename, formatstr)
    if as_grey and len(im.shape) == 3:
        if im.shape[2] == 1:
            return im.squeeze()
        # these are the values that wikipedia says are typical
        transform = np.array([ 0.30,  0.59,  0.11])
        return np.dot(im, transform)
    return im


def imread_multi(filename, formatstr=None):
    '''
    images = imread_multi(filename, formatstr={from filename})

    The file type is guessed from `filename`.

    Parameters
    ----------
    filename : str
        filename

    formatstr : str, optional
        file format. If ``None``, then it is derived from the filename.

    Returns
    -------
    images : list
    '''
    formatstr = _parse_formatstr(filename, formatstr, 'imread')
    return _imread.imread_multi(filename, formatstr)


def imsave(filename, array, formatstr=None):
    '''
    imsave(filename, array, formatstr={auto-detect})

    Writes `array` into file `filename`

    Parameters
    ----------
    filename : str
        path on file system
    array : ndarray-like
    formatstr: str, optional
        format string
    '''
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError('imread:imsave: only integer images are supported')
    array = np.ascontiguousarray(array)
    if formatstr is None:
        dot = filename.rfind('.')
        if dot < 0:
            raise ValueError('imread.imsave: dot not found in filename (%s)' % filename)
        formatstr = filename[dot+1:]
    _imread.imsave(filename, formatstr, array)
