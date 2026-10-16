=============================
How To Install Mahotas-imread
=============================

imread requires Python 3.11 or later and NumPy 2.0 or later.

With pixi (recommended)
-----------------------

`pixi <https://pixi.sh>`__ installs imread, together with the C libraries it
needs, from `conda-forge <https://conda-forge.org>`__::

    pixi init my-project
    cd my-project
    pixi add imread
    pixi run python -c "import imread; print(imread.__version__)"

In an existing pixi project, ``pixi add imread`` is all you need.

With pip
--------

::

    pip install imread

Binary wheels are provided for Linux (x86_64 and aarch64), macOS (Intel and
Apple Silicon, macOS 15 or later) and Windows (AMD64 and ARM64). On other
platforms, pip builds imread from source (see below).

From source
-----------

Building imread needs a C++ compiler, ``pkg-config`` and the development files
for libpng, libjpeg and libtiff. libwebp is optional: WebP support is built
only if it is found. pixi can provide all of these::

    git clone https://github.com/luispedro/imread
    cd imread
    pixi init --format pixi
    pixi add python numpy pip meson-python meson ninja pkg-config cxx-compiler \
        zlib libpng libjpeg-turbo libtiff libwebp
    pixi run pip install --no-build-isolation .

To build without WebP support, pass
``--config-settings=setup-args=-Dwebp=disabled`` to ``pip install``.

Development happens on `github <https://github.com/luispedro/imread>`__. Watch
out that *development versions are more likely to have problems*.

With nix
--------

::

    nix-env -iA nixpkgs.python3Packages.imread

or use the ``pkgs.python3Packages.imread`` object in your ``*.nix`` files.

Reading XCF files
-----------------

XCF (GIMP) files are read through the ``xcf2png`` utility, which is part of
xcftools. It must be installed separately and be in the ``PATH`` (for example,
``sudo apt-get install xcftools`` on Debian/Ubuntu). It is not available on
Windows.
