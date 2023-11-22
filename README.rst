================================
mahotas-imread: Read Image Files
================================

.. image:: https://api.travis-ci.com/luispedro/imread.png
   :target: https://travis-ci.com/luispedro/imread
.. image:: https://anaconda.org/conda-forge/imread/badges/license.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://anaconda.org/conda-forge/imread/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/imread
.. image:: https://anaconda.org/conda-forge/imread/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/imread

IO with images and numpy arrays.


Mahotas-imread is a simple module with a small number of functions:

imread
    Reads an image file
imread_multi
    Reads an image file with multiple images. Currently, TIFF and STK (a TIFF
    sub-based format) support this function.
imsave
    Writes an image file

Example (which uses `mahotas <https://luispedro.org/software/mahotas>`__ for
Gaussian filtering)::


    from imread import imread, imsave
    from mahotas import gaussian_filter
    lena = imread('lena.jpeg')

    lena = gaussian_filter(lena.astype(float), 4.)
    imsave('lena-filtered.jpeg', lena)


This grew out of frustration at current image loading solutions in Python, in
either my packages [`mahotas <https://mahotas.rtfd.io>`__] or packages from
others [`scikit-image <https://scikit-image.org/>`__, for example].

The relationship with numpy is very contained and this could be easily
repurposed to load images in other frameworks, even other programming
languages.

`Online documentation <https://imread.rtfd.io/>`__

Python versions 2.6, 2.7, 3.3+ are officially supported.

Python 3.2 (and earlier versions in the Python 3 series) are officially **not
supported**. Patches will be accepted if they do not mess up anything else, but
bug reports will not be considered as very high priority.

Citation
--------

.. _Citation:

If you use imread on a published publication, please cite the main `mahotas
<https://mahotas.rtfd.io>`__ paper (imread is a spin-off of mahotas):

    **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
    vision in Journal of Open Research Software, vol 1, 2013. [`DOI
    <https://dx.doi.org/10.5334/jors.ac>`__]


In Bibtex format::

    @article{mahotas,
        author = {Luis Pedro Coelho},
        title = {Mahotas: Open source software for scriptable computer vision},
        journal = {Journal of Open Research Software},
        year = {2013},
        doi = {https://dx.doi.org/10.5334/jors.ac},
        month = {July},
        volume = {1}
    }


Installation/Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest environment to install mahotas-imread is anaconda, through
`conda-forge <https://conda-forge.github.io>`__. Just use::

    conda config --add channels conda-forge
    conda install imread


To compile on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev
    sudo apt-get install xcftools

To compile on Mac::

    sudo port install libpng tiff webp

Either way, you can then compile with::

    python setup.py build

and install with::

    python setup.py install

On Windows, you can also just download a pre-built package from `C. Gohlke's
repository <https://www.lfd.uci.edu/~gohlke/pythonlibs/#imread>`__

On nix, you can use::

    nix-env -iA nixpkgs.python3Packages.imread

or use the ``pkgs.python3Packages.imread`` object in your ``*.nix`` files.

Links & Contacts
----------------

*Documentation*: `https://imread.readthedocs.io/ <https://imread.readthedocs.io/>`__

*Issue Tracker*: `github imread issues <https://github.com/luispedro/imread>`__

*Mailing List*: Use the `pythonvision mailing list
<https://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.

*Main Author & Maintainer*: `Luis Pedro Coelho <https://luispedro.org>`__ (follow on `twitter
<https://twitter.com/luispedrocoelho>`__ or `github
<https://github.com/luispedro>`__).

History
~~~~~~~

Version 0.7.5 (2023-11-22)
--------------------------
- Fix build issue (#43, patch by @carlosal1015)

Version 0.7.4 (2020-04-14)
--------------------------
- Add missing header files to distribution

Version 0.7.3 (2020-04-09)
--------------------------
- Add missing test data to distribution

Version 0.7.2 (2020-03-24)
--------------------------
- Fix several memory access bugs in parsers (reported by Robert Scott)

Version 0.7.1 (2019-05-09)
--------------------------
- Fix 16-bit RGB/RGBA TIFF write (patch by Tomi Aarnio)

Version 0.7.0 (2018-09-30)
--------------------------
- Add support for reading ImageJ ROIs


Version 0.6.1 (2018-02-15)
--------------------------
- Support pathlib paths as function arguments
- Fix 16 bit PNG write support (patch by Tomi Aarnio)


Version 0.6 (2016-09-21)
--------------------------
- Add `supports_format` function
- Make png compression level tunable when calling imsave
- Add imsave_multi
- Add partial support for reading PNG files in Gray+alpha format


Version 0.5.1 (2014-11-06)
--------------------------
- Improve tests to work after installation
- Fix compilation in MSVC (patch by Christoph Gohlke)


Version 0.5 (2014-10-16)
------------------------
- Add magic-number based format auto-detection
- Auto detect whether webp is installed
- Fix WebP reading (update to newer API)

Version 0.4 (2014-07-21)
------------------------
- Add configuration for TIFF saving
- Correctly save 16 bit PNG images
- Better error messages for JPEG


Version 0.3.2 (2013-10-06)
--------------------------
- Added imload*/imwrite synonyms as suggested by Thouis (Ray) Jones
- Options framework
- Allow user to specify JPEG quality when saving
- Fix loading of 16 bit PNG images

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
--------------------------
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

