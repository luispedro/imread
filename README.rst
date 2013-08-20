================================
mahotas-imread: Read Image Files
================================

Mahotas-imread is a simple module with a small number of functions:

imread
    Reads an image file
imread_multi
    Reads an image file with multiple images. Currently, TIFF and STK (a TIFF
    sub-based format) support this function.
imsave
    Writes an image file

Example (which uses `mahotas <http://luispedro.org/software/mahotas>`__ for
Gaussian filtering)::


    from imread import imread, imsave
    from mahotas import gaussian_filter
    lena = imread('lena.jpeg')

    lena = gaussian_filter(lena.astype(float), 4.)
    imsave('lena-filtered.jpeg', lena)


This grew out of frustration at current image loading solutions in Python, in
either my packages [`mahotas <http://mahotas.rtfd.org>`__] or packages from
others [`scikit-image <http://scikit-image.org/>`__, for example].

The relationship with numpy is very contained and this could be easily
repurposed to load images in other frameworks, even other programming
languages.

`Online documentation <http://packages.python.org/imread/>`__


Citation
--------

.. _Citation:

If you use imread on a published publication, please cite the main `mahotas
<http://mahotas.rtfd.org>`__ paper (imread is a spin-off of mahotas):

    **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
    vision in Journal of Open Research Software, vol 1, 2013. [`DOI
    <http://dx.doi.org/10.5334/jors.ac>`__]


In Bibtex format::

    @article{mahotas,
        author = {Luis Pedro Coelho},
        title = {Mahotas: Open source software for scriptable computer vision},
        journal = {Journal of Open Research Software},
        year = {2013},
        doi = {http://dx.doi.org/10.5334/jors.ac},
        month = {July},
        volume = {1}
    }


Dependencies
~~~~~~~~~~~~

To install on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev
    sudo apt-get install xcftools

To install on Mac::

    sudo port install libpng tiff webp


Bug Reports
~~~~~~~~~~~

Please report any bugs either on github or by email to luis@luispedro.org

If you are not sure of whether this is the correct behaviour, you can discuss
this on the
`pythonvision mailing list <https://groups.google.com/forum/?fromgroups#!forum/pythonvision>`__

If at all possible, include a small image as a test case.

Travis Build Status
~~~~~~~~~~~~~~~~~~~

.. image:: https://travis-ci.org/luispedro/imread.png
       :target: https://travis-ci.org/luispedro/imread

Python versions 2.6, 2.7 and 3.3 are officially supported.

Python 3.4 should also work (submit a bug report if it does not). Python 3.2
(and earlier versions in the Python 3 series) are officially **not supported**.

History
~~~~~~~

Version 0.3.1 (2013-06-20)
--------------------------
- Fix possible crash on error with TIFF
- Fix compilation on Windows (reported by Volker Hilsenstein)
- Make it easy to compile without WebP

Version 0.3.0 (2013-07-29)
--------------------------
- Support for reading from in-memory blobs
- Support for reading & writing TIFF metadata
- Add PHOTOMETRIC tag to TIFF (reported by Volker Hilsenstein)
- Support writing RGB TIFFs

Version 0.2.6 (2013-06-19)
--------------------------
- Fix hard crash when saving with non-existing file type
- Fix compilation on MacOS (patch by Alexander Bohn)
- Add ``verbose`` argument to tests.run()
- Better error when attempting to save floating point images

Version 0.2.5 (2012-10-29)
--------------------------
- Correctly accept uppercase extensions
- Python 3 support (patch by Christoph Gohlke [pull request 8 on github])
- Read 1-Bit PNGs
- Read simple BMPs (compression and many bit types not supported)
- More complete debug mode (export DEBUG=2 when building), more checks

Version 0.2.4 (2012-06-26)
-------------------------
- Add lzw.cpp to source distribution
- Support saving 16-bit TIFF
- Better Mac OS support (patch from Alexander Bohn)

Version 0.2.3 (2012-06-8)
-------------------------
- Fix imread_multi

Version 0.2.2 (2012-06-5)
-------------------------
- Add `formatstr` argument to imread
- Open files in binary mode on Windows (patch by Christoph Gohlke)
- Read-only support for LSM files
- Read-only support for XCF files (through `xcf2png`)
- Fix writing of non-contiguous images (at least PNG was affected)


Version 0.2.1 (2012-02-11)
--------------------------
- Add missing files to distribution

Version 0.2 (2012-03-19)
------------------------
- Compile on MSVC++ (Patches by Christoph Gohlke)
- Support for WebP
- Support for 1-bit TIFFs
- Better error message
- Support for multi-page TIFF reading
- Experimental read-only support for STK files


Version 0.1 (2012-02-28)
------------------------

- Support for PNG
- Support for TIFF
- Support for JPEG

