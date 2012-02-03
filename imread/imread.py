# -*- coding: utf-8 -*-
# Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

import numpy as np
import _imread

def imread(filename):
    '''
    im = imread(filename)

    The file type is guessed from `filename`.

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    im : ndarray
    '''
    dot = filename.rfind('.')
    if dot < 0:
        raise ValueError('imread.imread: Cannot determine file type from filename (%s)' % filename)
    formatstr = filename[1+dot:]
    return _imread.imread(filename, formatstr)


def imsave(filename, array):
    '''
    imsave(filename, array)

    Writes `array` into file `filename`

    Parameters
    ----------
    filename : str
        path on file system
    array : ndarray-like
    '''
    array = np.asanyarray(array)
    _imread.imsave(filename, array)
