=============
API Reference
=============

.. module:: imread

Reading
-------

.. autofunction:: imread

.. autofunction:: imread_multi

.. autofunction:: imread_from_blob

Writing
-------

.. autofunction:: imsave

.. autofunction:: imsave_multi

Formats
-------

.. autofunction:: detect_format

.. autofunction:: supports_format

Aliases
-------

These names are aliases for the functions above:

- ``imload`` is :func:`imread`
- ``imload_multi`` is :func:`imread_multi`
- ``imload_from_blob`` is :func:`imread_from_blob`
- ``imwrite`` is :func:`imsave`
- ``imwrite_multi`` is :func:`imsave_multi`

ImageJ ROIs
-----------

.. automodule:: imread.ijrois
   :members: read_roi, read_roi_zip
