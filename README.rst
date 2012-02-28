========================
imread: Read Image Files
========================

Simple module with two functions:

imread
    Reads an image file
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

History
~~~~~~~

Version 0.1 (2012-02-28)
------------------------

- Support for PNG
- Support for TIFF
- Support for JPEG

