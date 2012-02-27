# -*- coding: utf-8 -*-
# Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

import numpy as np
import _imread

def imread(filename, as_grey=False):
    '''
    im = imread(filename, as_grey=False)

    The file type is guessed from `filename`.

    Parameters
    ----------
    filename : str
        filename
    as_grey : boolean, optional
        Whether to convert to grey scale image (default: no)

    Returns
    -------
    im : ndarray
        The type of this array will depend on the contents of the file and of
        `as_grey`. Conversion from colour to grayscale will return a floating
        point image.
    '''
    im = _imread.imread(filename)
    if as_grey and len(im.shape) == 3:
        # these are the values that wikipedia says are typical
        transform = np.array([ 0.30,  0.59,  0.11])
        return np.dot(im, transform)
    return im


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
    array = np.asanyarray(array)
    if formatstr is None:
        dot = filename.rfind('.')
        if dot < 0:
            raise ValueError('imread.imsave: dot not found in filename (%s)' % filename)
        formatstr = filename[dot+1:]
    _imread.imsave(filename, formatstr, array)
