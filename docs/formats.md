=================
Formats Supported
=================

+---------------+------+-------+---------------+------------------------------+
| Format        | Read | Write | Since version | Format strings               |
+---------------+------+-------+---------------+------------------------------+
| PNG           |  Y   |   Y   |      0.1      | ``png``                      |
+---------------+------+-------+---------------+------------------------------+
| TIFF          |  Y   |   Y   |      0.1      | ``tiff``, ``tif``            |
+---------------+------+-------+---------------+------------------------------+
| JPEG          |  Y   |   Y   |      0.1      | ``jpeg``, ``jpg``            |
+---------------+------+-------+---------------+------------------------------+
| WebP          |  Y   |   N   |      0.2      | ``webp``                     |
+---------------+------+-------+---------------+------------------------------+
| BMP           |  Y   |   N   |      0.2.5    | ``bmp``                      |
+---------------+------+-------+---------------+------------------------------+
| STK           |  Y   |   N   |      0.2      | ``stk``                      |
+---------------+------+-------+---------------+------------------------------+
| LSM           |  Y   |   N   |      0.2.2    | ``lsm``                      |
+---------------+------+-------+---------------+------------------------------+
| XCF           |  Y   |   N   |      0.2.2    | ``xcf``                      |
+---------------+------+-------+---------------+------------------------------+

The format is taken from the file extension, unless it is given explicitly with
the ``formatstr`` argument. When reading, the extension is lowercased, so
``image.PNG`` works. When saving, it is not, so pass ``formatstr='png'`` if the
filename has an uppercase extension.

``imread_from_blob`` can detect PNG and JPEG data by their magic numbers. For
other formats, ``formatstr`` must be given.

``supports_format(formatstr)`` reports whether the compiled module handles a
format. It returns ``False`` for ``'webp'`` if imread was built without libwebp,
and always returns ``False`` for ``'xcf'``, which is not handled by the
compiled module (see below).

Data types and shapes
---------------------

Images are returned as NumPy arrays of shape ``(height, width)`` or
``(height, width, channels)``. Integer data is always returned as an unsigned
type (``uint8``, ``uint16`` or ``uint32``); 1-bit images are returned as
``bool``.

``imsave`` and ``imsave_multi`` accept only integer arrays (anything else
raises ``TypeError``). Use unsigned types: signed values are written as their
unsigned bit patterns (so ``-1`` in an ``int8`` array is saved as ``255``).

PNG
---

- **Reading:** 8- and 16-bit greyscale, RGB and RGBA, and 1-bit greyscale
  (returned as ``bool``). Paletted images with 8-bit indices are expanded to
  RGB, or to RGBA if they have transparency. Other paletted images, and 2- and
  4-bit greyscale images, are not supported. Greyscale+alpha images can only be
  read with ``opts={'strip_alpha': True}``, which drops the alpha channel.
- **Writing:** ``uint8`` and ``uint16`` arrays, either 2-dimensional
  (greyscale) or with 3 (RGB) or 4 (RGBA) channels.

TIFF
----

- **Reading:** unsigned integer images with 1, 8, 16 or 32 bits per sample and
  any number of channels. ``imread`` returns the first page and ``imread_multi``
  returns a list with all the pages.

  Floating-point and signed integer TIFFs are not supported: they are returned
  as unsigned integers holding the raw bits. Tiled TIFFs and TIFFs with
  separate colour planes are not supported either. Photometric interpretation
  is ignored, so palette images return the indices and "min-is-white" images
  are not inverted.
- **Writing:** 2-dimensional (greyscale) or multi-channel arrays of any
  unsigned integer type. ``imsave_multi`` writes several pages into one file.
  Images are LZW-compressed by default (see the options below).
- **Metadata:** ``imsave(..., metadata=b'...')`` writes the ImageDescription
  tag, and ``imread(..., return_metadata=True)`` returns it as ``bytes``
  (``b''`` if the tag is absent).

JPEG
----

- **Reading:** returns ``uint8`` arrays of shape ``(height, width, channels)``.
  Note that greyscale JPEGs are returned as ``(height, width, 1)``, unlike
  greyscale images in other formats. CMYK JPEGs are returned as 4 channels,
  without any colour conversion.
- **Writing:** ``uint8`` arrays only, either greyscale (2-dimensional or with a
  single channel) or RGB.

WebP
----

Read-only. Images are always returned as RGBA (``uint8``, 4 channels), even if
the file has no alpha channel. WebP support is only available if libwebp was
found when imread was built.

BMP
---

Read-only. Uncompressed bitmaps with 1, 4, 8, 16, 24 or 32 bits per pixel are
supported, in both bottom-up and top-down row order. Images are returned as RGB
(``uint8``, 3 channels). Paletted images (1, 4 and 8 bits) are expanded to RGB.
16- and 32-bit images may use bit-field colour masks; when they include an alpha
mask, the image is returned as RGBA (4 channels). Run-length encoded (RLE)
bitmaps are not supported.

STK
---

Read-only, and only with ``imread_multi``, which returns a list with one array
per plane.

LSM
---

Read-only (Zeiss confocal microscopy files). Only uncompressed files are
supported. The image is returned as a single 5-dimensional array, with the
planes, time points and channels as the first three dimensions.

XCF
---

Read-only, by converting the file (flattening all the layers) with the
``xcf2png`` utility, which is part of xcftools. ``xcf2png`` must be installed
separately and be in the ``PATH``. It is not available on Windows.

ImageJ ROIs
-----------

ImageJ regions of interest are not images, but can be read with the
``imread.ijrois`` module:

- ``read_roi(fileobj)`` reads a single ROI from an open (binary) file.
- ``read_roi_zip(filename)`` reads all the ROIs in a ZIP file, such as the ones
  saved by ImageJ's ROI Manager.

Each ROI is returned as an array of shape ``(n, 2)``, with one point per row,
in (y, x) order, with 1 subtracted from ImageJ's coordinates. Only freehand
ROIs are supported.

Options
-------

Some of the formats accept options, which are passed as a dictionary in the
``opts`` argument. Options that do not apply to the format are ignored.

Reading
~~~~~~~

strip_alpha
    (PNG) Whether to drop the alpha channel (default: False).

Writing
~~~~~~~

png:compression_level
    Compression level to use, from 0 (no compression) to 9. Setting it to 0 is
    discouraged, as 1 already provides a modicum of compression at no extra
    computational cost. The default is zlib's (6).

jpeg:quality
    An integer from 1 to 100 determining the quality used by the JPEG backend
    (default is the libjpeg default: 75). Values outside this range are
    clamped.

tiff:compress
    Whether to use (LZW) compression when saving TIFF (default: True).

tiff:horizontal-predictor
    Whether to use horizontal prediction in TIFF. This defaults to True for
    16-bit images, and to False for 8-bit images. This is because compressing
    16-bit images without horizontal prediction is often counterproductive
    (see `this discussion on the libtiff mailing list
    <https://web.archive.org/web/20190922141532/https://asmail.be/msg0055176395.html>`__).

tiff:XResolution, tiff:YResolution
    The resolution (pixels per unit), as an integer or a floating-point
    number. If only one of them is given, it is used for both.

tiff:XResolutionUnit
    The resolution unit, using the TIFF codes: 1 (none), 2 (inch) or
    3 (centimetre).
