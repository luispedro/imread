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
either my packages [`mahotas <http://mahotas.rtfd.org>`__] or packages from
others [`scikit-image <http://scikit-image.org>`__, for example].

The relationship with numpy is very contained and this could be easily
repurposed to load images in other frameworks, even other programming
languages.

Dependencies
~~~~~~~~~~~~

To install on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev

To install on Mac::

    sudo port install libpng tiff webp


Contents:

.. toctree::
   :maxdepth: 2

   readme
   formats
   non-python
   history

Bug Reports
~~~~~~~~~~~

Please report any bugs either on github or by email to luis@luispedro.org

If you are not sure of whether this is the correct behaviour, you can discuss
this on the
`pythonvision mailing list <https://groups.google.com/forum/?fromgroups#!forum/pythonvision>`__

If at all possible, include a small image as a test case.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

