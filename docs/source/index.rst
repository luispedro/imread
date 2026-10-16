Welcome to imread's documentation!
==================================

Imread is a very simple library. It has three functions:

imread
    Reads an image from disk
imread_multi
    Reads multiple images from disk (only for file formats that support
    multiple images)
imwrite
    Saves an image to disk

That's it.

It is meant to be used with a computer vision & image processing package, such
as:

- `mahotas <https://luispedro.org/software/mahotas>`__
- `scikit-image <https://scikit-image.org/>`__
- `OpenCV <https://opencv.org/>`__


This grew out of frustration at current image loading solutions in Python, in
either my packages [`mahotas <https://mahotas.rtfd.io>`__] or packages from
others [`scikit-image <https://scikit-image.org>`__, for example].

The relationship with NumPy is very contained, so this could easily be
repurposed to load images in other frameworks, or even in other programming
languages.

Citation
--------

This package is an offshoot of mahotas. As it does not currently have its own
publication, you are asked to cite the parent package. If you use
mahotas-imread in a scientific publication, please cite:

    **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
    vision in Journal of Open Research Software, vol 1, 2013. [`DOI
    <https://doi.org/10.5334/jors.ac>`__]


In BibTeX format::

    @article{mahotas,
        author = {Luis Pedro Coelho},
        title = {Mahotas: Open source software for scriptable computer vision},
        journal = {Journal of Open Research Software},
        year = {2013},
        doi = {10.5334/jors.ac},
        month = {July},
        volume = {1}
    }


Installation
------------

The recommended way to install imread is with `pixi <https://pixi.sh>`__::

    pixi add imread

Binary wheels are also available with ``pip install imread``. See
:doc:`install` for all the options, including building from source.


Contents:

.. toctree::
   :maxdepth: 2

   readme
   install
   formats
   non-python
   history

Bug Reports
-----------

Please report any bugs either on `GitHub
<https://github.com/luispedro/imread>`__ or by email to luis@luispedro.org.

If you have a test case where you are not sure whether imread is behaving
correctly, you can discuss it on the `pythonvision mailing list
<https://groups.google.com/forum/?fromgroups#!forum/pythonvision>`__.

If at all possible, include a small image as a test case.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

