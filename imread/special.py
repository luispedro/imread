# Copyright (C) 2012-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING.MIT file)

import imread

def readxcf(xcf_filename, formatstr, _flags):
    '''Read an XCF file, flattening all the layers.

    Depends on the ``xcf2png`` utility being available.

    Parameters
    ----------
    xcf_filename : str
        Path to the file.
    formatstr : str
        Format name. Must be ``'xcf'``.
    _flags : dict
        Reader options. These are ignored, but the argument is needed because
        all readers are called as ``reader(filename, formatstr, opts)``.

    Returns
    -------
    im : ndarray
        The image.
    meta : None
        The metadata (as for PNG, always None).
    '''
    from os import system, unlink
    from tempfile import NamedTemporaryFile
    if formatstr != 'xcf':
        raise ValueError('imread.imread.readxcf: Format string must be \'xcf\'')

    N = NamedTemporaryFile(suffix='.png')
    output = system('xcf2png %s >%s' % (xcf_filename,N.name))
    if output:
        raise OSError('imread.readxcf: xcf format is only supported through the xcf2png utility, which imread could not run')
    return imread.imread(N.name, return_metadata=True)

special = {
    'xcf' : readxcf,
}
