Welcome to imread's documentation!
==================================

Imread is a very simple libray. It has three functions

imread
    Reads an image from disk
imread_multi
    Reads multiple images from disk (only for file formats that support
    multiple images)
imwrite
    Save an image to disk

That's it.

This needs to be used with a a computer vision & image processing packages:

- `mahotas <http://luispedro.org/software/mahotas>`__
- `scikit-image <http://scikit-image.org/>`__
- `OpenCV <http://opencv.willowgarage.com/wiki/>`__


This grew out of frustration at current image loading solutions in Python, in
either my packages [`mahotas <http://mahotas.rtfd.io>`__] or packages from
others [`scikit-image <http://scikit-image.org>`__, for example].

The relationship with numpy is very contained and this could be easily
repurposed to load images in other frameworks, even other programming
languages.

Citation
--------

This package is an off-shoot of mahotas. As it, currently, does not have its
own publication, so you are asked to cite the mother package: If you use
imread-mahotas on a scientific publication, please cite:

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


INSTALL
~~~~~~~

On Windows, you can also just download a pre-built package from `C. Gohlke's
repository <http://www.lfd.uci.edu/~gohlke/pythonlibs/#imread>`__

To compile on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev
    sudo apt-get install xcftools

To compile on Mac::

    sudo port install libpng tiff webp

Either way, you can then install::

    pip install imread


Contents:

.. toctree::
   :maxdepth: 2

   readme
   install
   formats
   non-python
   history

Bug Reports
~~~~~~~~~~~

Please report any bugs either on `github
<http://github.com/luispedro/imread>`__ or by email to luis@luispedro.org

If you have a test case where are not sure of whether imread is behaving
correctly, you can discuss this on the `pythonvision mailing list
<https://groups.google.com/forum/?fromgroups#!forum/pythonvision>`__

If at all possible, include a small image as a test case.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

