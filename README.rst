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

Dependencies
~~~~~~~~~~~~

To install on debian/ubuntu::

    sudo apt-get install libpng12-dev libtiff4-dev

