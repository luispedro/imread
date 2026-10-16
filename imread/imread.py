# Copyright (C) 2012-2016, Luis Pedro Coelho <luis@luispedro.org>
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
    raise ValueError('imread.{0}: Could not identify format from filename: `{1}`'.format(funcname,filename))

def _as_grey(im, as_grey):
    if not as_grey or len(im.shape) != 3: return im
    if im.shape[2] == 1:
        return im.squeeze()
    # these are the values that wikipedia says are typical
    transform = np.array([ 0.30,  0.59,  0.11])
    return np.dot(im, transform)

def _norm_path(path):
    import os
    if hasattr(os, 'PathLike') and isinstance(path, os.PathLike):
        return str(path)
    return path

def imread(filename, as_grey=False, formatstr=None, return_metadata=False, opts=None):
    '''
    im = imread(filename, as_grey=False, formatstr={filename extension}, return_metadata=False)
    im,meta = imread(filename, as_grey=False, formatstr={filename extension}, return_metadata=True)

    Read an image into an ndarray from a file.

    Parameters
    ----------
    filename : str or os.PathLike
        filename
    as_grey : boolean, optional
        Whether to convert to a greyscale image (default: False)
    formatstr : str, optional
        Format name. This is typically the same as the extension of the file
        and inferred from there. However, if you have a file whose extension
        does not correspond to the format, you can pass it explicitly.
    return_metadata : bool, optional
        Whether to return metadata (default: False)
    opts : dict, optional
        Other options. Support will depend on the backend. Currently used:

            strip_alpha: bool
                Whether to strip the alpha channel.

    Returns
    -------
    im : ndarray
        The type of this array will depend on the contents of the file and of
        `as_grey`. Conversion from colour to greyscale will return a floating
        point image.
    meta : bytes or None
        The metadata in the file. Only returned if ``return_metadata``. For
        TIFF, this is the ImageDescription tag as ``bytes`` (``b''`` if the tag
        is absent). For other formats, it is ``None``.

    See Also
    --------
    imread_from_blob : function
        Read from an in-memory blob
    '''
    if opts is None:
        opts = {}
    filename = _norm_path(filename)
    formatstr = _parse_formatstr(filename, formatstr, 'imread')
    reader = special.get(formatstr, _imread.imread)
    imdata,meta = reader(filename, formatstr, opts)
    imdata = _as_grey(imdata, as_grey)
    if return_metadata:
        return imdata, meta
    return imdata

imload = imread


def imread_from_blob(blob, formatstr=None, as_grey=False, return_metadata=False, opts=None):
    '''
    imdata = imread_from_blob(blob, formatstr=None, as_grey=False, return_metadata=False)
    imdata,metadata = imread_from_blob(blob, formatstr=None, as_grey=False, return_metadata=True)

    Read an image into an ndarray from an in-memory blob.

    Parameters
    ----------
    blob : bytes
        Input data
    formatstr : str, optional
        Format name. This is the file extension typically associated with this
        format. If None is given, then ``detect_format`` is used to auto-detect
        the format. Auto-detection only works for PNG and JPEG, so
        ``formatstr`` must be given for other formats.
    as_grey : boolean, optional
        Whether to convert to a greyscale image (default: False)
    return_metadata : bool, optional
        Whether to return metadata (default: False)
    opts : dict, optional
        Other options. Support will depend on the backend. Currently used:

            strip_alpha: bool
                Whether to strip the alpha channel.

    Returns
    -------
    im : ndarray
        The type of this array will depend on the contents of the file and of
        `as_grey`. Conversion from colour to greyscale will return a floating
        point image.
    meta : bytes or None
        The metadata in the file. Only returned if ``return_metadata``. For
        TIFF, this is the ImageDescription tag as ``bytes`` (``b''`` if the tag
        is absent). For other formats, it is ``None``.

    See Also
    --------
    imread : function
        Read from a file on disk
    '''
    if opts is None:
        opts = {}
    reader = _imread.imread_from_blob
    if formatstr is None:
        formatstr = detect_format(blob, is_blob=True)
    imdata,meta = reader(blob, formatstr, opts)
    imdata = _as_grey(imdata, as_grey)
    if return_metadata:
        return imdata,meta
    return imdata

imload_from_blob = imread_from_blob

def imread_multi(filename, formatstr=None):
    '''
    images = imread_multi(filename, formatstr={from filename})

    The file type is guessed from `filename`.

    Parameters
    ----------
    filename : str or os.PathLike
        filename

    formatstr : str, optional
        File format. If ``None``, then it is derived from the filename.

    Returns
    -------
    images : list
    '''
    filename = _norm_path(filename)
    formatstr = _parse_formatstr(filename, formatstr, 'imread')
    return _imread.imread_multi(filename, formatstr, '')

imload_multi = imread_multi


def _as_native_contiguous(array):
    # The C++ code reads the raw buffer, so it must be in native byte order
    array = array.astype(array.dtype.newbyteorder('='), copy=False)
    return np.ascontiguousarray(array)


def imsave(filename, array, formatstr=None, metadata=None, opts=None):
    '''
    imsave(filename, array, formatstr={auto-detect}, metadata={None}, opts={})

    Writes `array` into file `filename`.

    Parameters
    ----------
    filename : str or os.PathLike
        Path on the file system
    array : ndarray-like
    formatstr : str, optional
        Format string
    metadata : bytes, optional
        Metadata to write to the file. Note that not all formats support writing
        metadata.
    opts : dict, optional
        This is a dictionary of options. Any non-applicable option is typically
        silently ignored. Currently, the following options are accepted:

        jpeg:quality
            An integer 1-100 determining the quality used by the JPEG backend
            (default is the libjpeg default: 75).

        tiff:compress
            Whether to use compression when saving TIFF (default: True).

        tiff:horizontal-predictor
            Whether to use horizontal prediction in TIFF. This defaults to True
            for 16-bit images, and to False for 8-bit images. This is because
            compressing 16-bit images without horizontal prediction is often
            counterproductive (see https://web.archive.org/web/20190922141532/https://asmail.be/msg0055176395.html).

        png:compression_level
            Compression level to use, from 0 (no compression) to 9. Setting
            this parameter to 0 is discouraged, as setting it to 1 already
            provides a modicum of compression at no extra computational cost.

    '''
    if opts is None:
        opts = {}
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError('imread.imsave: only integer images are supported')
    array = _as_native_contiguous(array)
    filename = _norm_path(filename)
    if formatstr is None:
        dot = filename.rfind('.')
        if dot < 0:
            raise ValueError('imread.imsave: dot not found in filename ({})'.format(filename))
        formatstr = filename[dot+1:]
    if metadata is not None:
        opts['metadata'] = metadata
    _imread.imsave(filename, formatstr, array, opts)

def imsave_multi(filename, arrays, formatstr=None, opts=None):
    '''Saves multiple arrays into a single file.

    Only certain formats (in the current version of imread, only TIFF) support
    writing multiple outputs.

    Parameters
    ----------
    filename : str or os.PathLike
        Path on the file system
    arrays : list of ndarray-like
    formatstr : str, optional
        Format string
    metadata : bytes, optional
        Metadata to write to the file. Note that not all formats support writing
        metadata.
    opts : dict, optional
        This is a dictionary of options. Any non-applicable option is typically
        silently ignored. Currently, the following options are accepted:

        jpeg:quality
            An integer 1-100 determining the quality used by the JPEG backend
            (default is the libjpeg default: 75).

        tiff:compress
            Whether to use compression when saving TIFF (default: True).

        tiff:horizontal-predictor
            Whether to use horizontal prediction in TIFF. This defaults to True
            for 16-bit images, and to False for 8-bit images. This is because
            compressing 16-bit images without horizontal prediction is often
            counterproductive (see https://web.archive.org/web/20190922141532/https://asmail.be/msg0055176395.html).

        png:compression_level
            Compression level to use, from 0 (no compression) to 9. Setting
            this parameter to 0 is discouraged, as setting it to 1 already
            provides a modicum of compression at no extra computational cost.
    '''
    if opts is None:
        opts = {}
    for array in arrays:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError('imread.imsave: only integer images are supported')
    arrays = [_as_native_contiguous(array) for array in arrays]
    filename = _norm_path(filename)
    if formatstr is None:
        dot = filename.rfind('.')
        if dot < 0:
            raise ValueError('imread.imsave_multi: dot not found in filename ({})'.format(filename))
        formatstr = filename[dot+1:]
    _imread.imsave_multi(filename, formatstr, arrays, opts)

imwrite = imsave
imwrite_multi = imsave_multi


def detect_format(filename_or_blob, is_blob=False):
    '''
    Detect format using magic numbers.

    Note that this function does not check that the data is valid; it only
    checks magic numbers. Not all formats can be autodetected.

    Parameters
    ----------
    filename_or_blob : str, os.PathLike, or bytes
    is_blob : boolean, optional
        If true, then filename_or_blob is interpreted as a blob; otherwise,
        it is interpreted as a filename.

    Returns
    -------
    format : str or None
        Representation of the format if detected. When no format matches, None
        is returned.
    '''
    if not is_blob:
        filename_or_blob = _norm_path(filename_or_blob)
    return _imread.detect_format(filename_or_blob, bool(is_blob))


def supports_format(formatstr):
    '''Query whether the format is supported by imread.

    Parameters
    ----------
    formatstr : str
        Format string (for example, 'png')

    Returns
    -------
    is_supported : bool
    '''
    return _imread.supports_format(formatstr)
