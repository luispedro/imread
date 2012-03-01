==============================
Using imread outside of Python
==============================

Actually, **imread is not a Python library**. It's really a C++ library with
very good Python/numpy bindings. But you can easily use it from other
languages (including C++, of course) by adapting your image representation to
the imread interface.

The Image and the ImageFactory interfaces
-----------------------------------------

You need to support the ``Image`` interface::

    class Image {
        public:
            // number of dimensions
            virtual int ndims() const = 0;

            // size of dimension d
            virtual int dim(int d) const = 0;

            // get a pointer to row `r`
            virtual void* rowp(int r) = 0;
    };

It is assumed that an RGB image is represented as *H x W x 3* and RGBA as *H x
W x 4*.

You also need to support the ``ImageFactory`` interface::

    class ImageFactory {
        public:
            Image* create(int nbits, int w, int h, int d);
    };

The ``create`` methods create 2 or 3-dimensional images with ``nbits`` per pixel.

See the numpy interface in the source code for inspiration.

