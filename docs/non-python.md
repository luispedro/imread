==============================
Using imread outside of Python
==============================

The image formats are implemented in C++, in ``imread/lib/``. That code does
not depend on Python or NumPy: the NumPy-specific parts are in
``lib/numpy.h``, ``lib/numpy.cpp`` and ``_imread.cpp``. So you can use imread
from other languages (including C++, of course) by adapting your image
representation to the imread interfaces.

All the declarations below are in ``lib/base.h``.

The Image and the ImageFactory interfaces
-----------------------------------------

Images are represented by the ``Image`` interface::

    class Image {
        public:
            virtual ~Image();

            // a copy of this image
            virtual std::unique_ptr<Image> clone() const = 0;

            // get a pointer to row `r`
            virtual void* rowp(int r) = 0;

            // number of bits per sample (1, 8, 16, ...)
            virtual int nbits() const = 0;

            // number of dimensions
            virtual int ndims() const = 0;

            // size of dimension d
            virtual int dim(int d) const = 0;
    };

Dimension 0 is the height, and the rest of the image must be laid out
contiguously (in row-major order) after each ``rowp(r)``. An RGB image is
represented as *H x W x 3* and RGBA as *H x W x 4*. Some formats use more
dimensions: LSM images, for example, are 5-dimensional.

When reading, the format code asks an ``ImageFactory`` to allocate the image
and then fills it in, row by row::

    class ImageFactory {
        public:
            virtual ~ImageFactory();
            virtual std::unique_ptr<Image>
                create(int nbits, int d0, int d1, int d2, int d3=-1, int d4=-1) = 0;
    };

``d0`` is the height and ``d1`` the width. Unused trailing dimensions are
passed as ``-1``, so a greyscale image is ``create(8, h, w, -1)`` and an RGB
image is ``create(8, h, w, 3)``.

When ``nbits`` is 1, the rows are filled with *packed* bits (8 pixels per byte,
most significant bit first), but each row should still be allocated with one
byte per pixel. ``NumpyImage::finalize()`` (in ``lib/numpy.cpp``) shows how to
expand the bits in place.

To attach metadata (such as the TIFF ImageDescription) to an image, have your
image class also derive from ``ImageWithMetadata``. The TIFF reader stores the
metadata with ``set_meta()``, and you can get it back with ``get_meta()``.

Formats
-------

Each format is a subclass of ``ImageFormat``::

    class ImageFormat {
        public:
            virtual bool can_read() const;
            virtual bool can_read_multi() const;
            virtual bool can_write() const;
            virtual bool can_write_multi() const;
            virtual bool can_write_metadata() const;

            virtual std::unique_ptr<Image>
                read(byte_source* src, ImageFactory* factory, const options_map& opts);
            virtual std::unique_ptr<image_list>
                read_multi(byte_source* src, ImageFactory* factory, const options_map& opts);
            virtual void write(Image* input, byte_sink* output, const options_map& opts);
            virtual void write_multi(image_list* input, byte_sink* output, const options_map& opts);
    };

``get_format(formatstr)`` (declared in ``lib/formats.h``) returns the
``ImageFormat`` for a format string such as ``"png"`` or ``"tif"``, or a null
pointer if the format is unknown. ``magic_format(src)`` guesses the format from
the first bytes of the input (only PNG and JPEG are detected).

``options_map`` is a ``std::map<std::string, number_or_string>`` holding the
options described in :doc:`formats` (``"jpeg:quality"``, ``"tiff:compress"``,
...). ``image_list`` is a list of images, which owns its members.

Errors are reported by throwing the exceptions declared in ``lib/errors.h``
(``CannotReadError``, ``CannotWriteError``, ``NotImplementedError``, ...), all
of which derive from ``std::exception``.

Input and output
----------------

Formats read from a ``byte_source`` and write to a ``byte_sink``. Two
implementations are provided:

- ``fd_source_sink`` (``lib/file.h``) wraps a file descriptor, which it closes
  when destroyed. It can be used both as a source and as a sink.
- ``memory_source`` (``lib/memory.h``) reads from a buffer in memory. It does
  not copy the data, which must outlive it.

To read from or write to anything else, implement these interfaces. A
``byte_source`` must implement ``read(byte* buffer, size_t n)``, returning the
number of bytes read; a ``byte_sink`` must implement
``write(const byte* buffer, size_t n)``, returning the number of bytes
written. Some formats (TIFF, BMP and LSM) also need to seek, which requires
implementing ``can_seek()``, ``seek_absolute()``, ``seek_relative()`` and
``seek_end()``.

Example
-------

This program reads a PNG file into a ``std::vector`` and saves it as an
uncompressed TIFF::

    #include "base.h"
    #include "formats.h"
    #include "file.h"

    #include <fcntl.h>
    #include <iostream>
    #include <vector>

    // An image stored in a std::vector, in row-major order
    class VectorImage : public Image {
        public:
            VectorImage(int nbits, std::vector<int> dims)
                :nbits_(nbits)
                ,dims_(dims)
                ,data_(row_size() * dims_.at(0))
                { }

            std::unique_ptr<Image> clone() const override {
                return std::unique_ptr<Image>(new VectorImage(*this));
            }
            int nbits() const override { return nbits_; }
            int ndims() const override { return dims_.size(); }
            int dim(int d) const override { return dims_.at(d); }
            void* rowp(int r) override { return &data_[r * row_size()]; }

        private:
            size_t row_size() const {
                size_t n = nbytes();
                for (size_t i = 1; i != dims_.size(); ++i) n *= dims_[i];
                return n;
            }
            int nbits_;
            std::vector<int> dims_;
            std::vector<byte> data_;
    };

    class VectorFactory : public ImageFactory {
        public:
            std::unique_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) override {
                std::vector<int> dims;
                for (int d : {d0, d1, d2, d3, d4}) {
                    if (d == -1) break;
                    dims.push_back(d);
                }
                return std::unique_ptr<Image>(new VectorImage(nbits, dims));
            }
    };

    int main(int argc, char* argv[]) {
        if (argc != 3) {
            std::cerr << "usage: " << argv[0] << " input.png output.tiff\n";
            return 1;
        }
        const int in_fd = ::open(argv[1], O_RDONLY);
        const int out_fd = ::open(argv[2], O_CREAT|O_RDWR|O_TRUNC, 0644);
        if (in_fd < 0 || out_fd < 0) {
            std::cerr << "Could not open files\n";
            return 1;
        }
        try {
            VectorFactory factory;
            options_map opts;

            fd_source_sink input(in_fd);
            std::unique_ptr<Image> im = get_format("png")->read(&input, &factory, opts);
            std::cout << "Read " << im->dim(0) << " x " << im->dim(1)
                      << " image (" << im->nbits() << " bits)\n";

            fd_source_sink output(out_fd);
            opts["tiff:compress"] = number_or_string(0);
            get_format("tiff")->write(im.get(), &output, opts);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << '\n';
            return 1;
        }
    }

Building
--------

The meson build only produces the Python extension module, so to use imread
from C++ you need to compile its sources into your own project. Compile all the
``.cpp`` files in ``imread/lib/`` except ``numpy.cpp`` (which needs Python and
NumPy) and ``lzw.cpp`` (which ``_lsm.cpp`` includes directly), and link against
libpng, libjpeg and libtiff, plus libwebp for WebP support. For example, to
build the program above (C++11 or later is needed)::

    c++ -std=c++17 -Iimread/lib example.cpp \
        imread/lib/{formats,_bmp,_jpeg,_lsm,_png,_tiff,_webp}.cpp \
        $(pkg-config --cflags --libs libpng libjpeg libtiff-4 libwebp) \
        -o example

To build without WebP, leave out ``_webp.cpp`` and ``libwebp``, and add
``-DIMREAD_EXCLUDE_WEBP``.
