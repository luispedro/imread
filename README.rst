========================
imread: Read Image Files
========================

Simple module with a small number of functions:

imread
    Reads an image file
imread_multi
    Reads an image file with multiple images. Currently, TIFF and STK (a TIFF
    sub-based format) support this function.
imsave
    Writes an image file

This grew out of frustration at current image loading solutions in Python, in
either my packages [mahotas] or packages from others [scikits.image, for
example].

The relationship with numpy is very contained and this could be easily
repurposed to load images in other frameworks, even other programming
languages.


Dependencies
~~~~~~~~~~~~

To install on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev

To install on Mac::

    sudo port install libpng tiff webp

Bug Reports
~~~~~~~~~~~

Please report any bugs either on github or by email to luis@luispedro.org

If you are not sure of whether this is the correct behaviour, you can discuss
this on the
`pythonvision mailing list <https://groups.google.com/forum/?fromgroups#!forum/pythonvision>`__

If at all possible, include a small image as a test case.

History
~~~~~~~

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

