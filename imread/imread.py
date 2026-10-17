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
    '''Read an image from a file.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file.
    as_grey : bool, optional
        Whether to convert the image to greyscale (default: False).
    formatstr : str, optional
        Format name (see :doc:`formats`). By default, the format is taken from
        the lowercased file extension, so ``image.PNG`` is read as PNG. Aliases
        such as ``jpg`` and ``tif`` are accepted. Pass it explicitly if the
        extension does not match the format.
    return_metadata : bool, optional
        Whether to also return the metadata (default: False).
    opts : dict, optional
        Options for the reader. Options that do not apply to the format are
        ignored. Currently, only one is used:

        strip_alpha : bool
            (PNG only) Whether to drop the alpha channel (default: False).

    Returns
    -------
    im : ndarray
        The image, of shape ``(height, width)`` or ``(height, width,
        channels)``. Integer data is always returned as an unsigned type
        (``uint8``, ``uint16`` or ``uint32``), and 1-bit images as ``bool``.
        See :doc:`formats` for the details of each format.

        With ``as_grey=True``, RGB images are converted with the weights 0.30,
        0.59 and 0.11 and returned as ``float64``. Single-channel images are
        returned as 2-dimensional arrays of their original type.
    meta : bytes or None
        The metadata in the file. Only returned if ``return_metadata``. For
        TIFF, this is the ImageDescription tag as ``bytes`` (``b''`` if the tag
        is absent). For other formats, it is ``None``.

    See Also
    --------
    imread_from_blob : Read an image from an in-memory blob.
    imread_multi : Read all the images in a multi-image file.
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
    '''Read an image from an in-memory blob.

    Parameters
    ----------
    blob : bytes
        The contents of an image file.
    formatstr : str, optional
        Format name (see :doc:`formats`). If it is not given,
        :func:`detect_format` is used to detect the format. Detection only
        works for PNG and JPEG, so ``formatstr`` must be given for other
        formats.
    as_grey : bool, optional
        Whether to convert the image to greyscale (default: False).
    return_metadata : bool, optional
        Whether to also return the metadata (default: False).
    opts : dict, optional
        Options for the reader. Options that do not apply to the format are
        ignored. Currently, only one is used:

        strip_alpha : bool
            (PNG only) Whether to drop the alpha channel (default: False).

    Returns
    -------
    im : ndarray
        The image, of shape ``(height, width)`` or ``(height, width,
        channels)``. Integer data is always returned as an unsigned type
        (``uint8``, ``uint16`` or ``uint32``), and 1-bit images as ``bool``.
        See :doc:`formats` for the details of each format.

        With ``as_grey=True``, RGB images are converted with the weights 0.30,
        0.59 and 0.11 and returned as ``float64``. Single-channel images are
        returned as 2-dimensional arrays of their original type.
    meta : bytes or None
        The metadata in the file. Only returned if ``return_metadata``. For
        TIFF, this is the ImageDescription tag as ``bytes`` (``b''`` if the tag
        is absent). For other formats, it is ``None``.

    See Also
    --------
    imread : Read an image from a file.
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
    '''Read all the images in a multi-image file.

    Only TIFF (all the pages) and STK (all the planes) are supported. Unlike
    :func:`imread`, this function returns no metadata and takes no options.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file.
    formatstr : str, optional
        Format name: ``tiff`` (or ``tif``) or ``stk``. By default, it is taken
        from the lowercased file extension.

    Returns
    -------
    images : list of ndarray
        One array per image in the file. See :func:`imread` for the types and
        shapes.

    See Also
    --------
    imread : Read a single image.
    imsave_multi : Write several images into a single file.
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
    '''Write an image to a file.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file.
    array : ndarray
        The image. It must have an integer type, and should be unsigned
        (signed values are written as their unsigned bit patterns). The
        accepted types and shapes depend on the format (see :doc:`formats`).
    formatstr : str, optional
        Format name (see :doc:`formats`). By default, it is taken from the file
        extension. Unlike in :func:`imread`, the extension is not lowercased,
        so pass ``formatstr`` if it is in uppercase.
    metadata : bytes, optional
        Metadata to write to the file. Only TIFF supports this (it is written
        as the ImageDescription tag); for other formats, it is ignored.
    opts : dict, optional
        Options for the writer. Options that do not apply to the format are
        ignored. Currently, the following ones are used:

        png:compression_level
            Compression level to use, from 0 (no compression) to 9. Setting it
            to 0 is discouraged, as 1 already provides a modicum of compression
            at no extra computational cost. The default is zlib's (6).

        jpeg:quality
            An integer from 1 to 100 determining the quality used by the JPEG
            backend (default is the libjpeg default: 75).

        tiff:compress
            Whether to use (LZW) compression when saving TIFF (default: True).

        tiff:horizontal-predictor
            Whether to use horizontal prediction in TIFF. This defaults to True
            for 16-bit images, and to False for 8-bit images. This is because
            compressing 16-bit images without horizontal prediction is often
            counterproductive (see `this discussion on the libtiff mailing list
            <https://web.archive.org/web/20190922141532/https://asmail.be/msg0055176395.html>`__).

        tiff:XResolution, tiff:YResolution
            The resolution (pixels per unit), as an integer or a floating-point
            number. If only one of them is given, it is used for both.

        tiff:XResolutionUnit
            The resolution unit, using the TIFF codes: 1 (none), 2 (inch) or
            3 (centimetre).

    Raises
    ------
    TypeError
        If ``array`` does not have an integer type.

    See Also
    --------
    imsave_multi : Write several images into a single file.
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
    '''Write several images into a single file.

    Only TIFF supports this: each array is written as one page.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file.
    arrays : list of ndarray
        The images. As in :func:`imsave`, they must have an integer type and
        should be unsigned (see :doc:`formats`).
    formatstr : str, optional
        Format name: ``tiff`` (or ``tif``). By default, it is taken from the
        file extension (which is not lowercased).
    opts : dict, optional
        Options for the writer. Options that do not apply to the format are
        ignored. Currently, the following ones are used:

        metadata
            Metadata (``bytes``) to write as the ImageDescription tag of each
            page. Unlike :func:`imsave`, this function has no ``metadata``
            argument.

        tiff:compress
            Whether to use (LZW) compression when saving TIFF (default: True).

        tiff:horizontal-predictor
            Whether to use horizontal prediction in TIFF. This defaults to True
            for 16-bit images, and to False for 8-bit images. This is because
            compressing 16-bit images without horizontal prediction is often
            counterproductive (see `this discussion on the libtiff mailing list
            <https://web.archive.org/web/20190922141532/https://asmail.be/msg0055176395.html>`__).

        tiff:XResolution, tiff:YResolution
            The resolution (pixels per unit), as an integer or a floating-point
            number. If only one of them is given, it is used for both.

        tiff:XResolutionUnit
            The resolution unit, using the TIFF codes: 1 (none), 2 (inch) or
            3 (centimetre).

    Raises
    ------
    TypeError
        If any of the arrays does not have an integer type.

    See Also
    --------
    imsave : Write a single image.
    imread_multi : Read all the images in a multi-image file.
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
    '''Detect the format of a file or blob from its magic number.

    Only PNG and JPEG can be detected. This function does not check that the
    data is valid; it only checks the magic number.

    Parameters
    ----------
    filename_or_blob : str, os.PathLike or bytes
        Path to the file, or its contents.
    is_blob : bool, optional
        If true, ``filename_or_blob`` is interpreted as the contents of a file
        (a blob); otherwise, it is interpreted as a filename (default: False).

    Returns
    -------
    format : str or None
        ``'png'`` or ``'jpeg'``, or None if neither matches.
    '''
    if not is_blob:
        filename_or_blob = _norm_path(filename_or_blob)
    return _imread.detect_format(filename_or_blob, bool(is_blob))


def supports_format(formatstr):
    '''Check whether a format is supported by the compiled module.

    This reports what the compiled module handles, so it returns False for
    ``'webp'`` if imread was built without libwebp, and always returns False
    for ``'xcf'``, which is read through the external ``xcf2png`` utility (see
    :doc:`formats`).

    Parameters
    ----------
    formatstr : str
        Format name (for example, ``'png'``).

    Returns
    -------
    is_supported : bool
    '''
    return _imread.supports_format(formatstr)
