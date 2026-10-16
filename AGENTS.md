# AGENTS.md

This file provides guidance to AI coding agents (such as Claude Code) when working with code in this repository.

## Overview

`imread` (mahotas-imread) is a Python library for reading and writing images as NumPy arrays. A thin Python layer wraps one C++ extension module, `imread._imread`, which links against libpng, libjpeg, libtiff and, optionally, libwebp.

## Build and test

You must build the extension in place before you can run the tests:

```bash
make debug     # DEBUG=2 python setup.py build --build-lib=.  (asserts on, _GLIBCXX_DEBUG; this is what CI uses)
make fast      # optimized in-place build
make tests     # make debug, then python -m pytest
make clean     # remove build/ and the in-place .so files
```

Run a single test:

```bash
python -m pytest imread/tests/test_png.py::test_asym -v
```

Build notes:
- System dependencies are the libpng, libjpeg, libtiff and libwebp dev headers. CI installs `jpeg libpng libwebp libtiff gcc_linux-64 gxx_linux-64 pytest` into a micromamba environment.
- WebP support is detected with `pkg-config --exists libwebp`. Setting `EXCLUDE_WEBP=1` forces it off and defines `IMREAD_EXCLUDE_WEBP`, so WebP code must stay behind that macro.
- `DEBUG=1` removes `NDEBUG`. `DEBUG=2` also defines `_GLIBCXX_DEBUG`, which turns on the bounds-checked `read`/`write` array overloads in `lib/base.h`.
- After you add a `.cpp` file to `imread/lib/`, list it in `extensions` in `setup.py`. The exception is `lzw.cpp`, which `_lsm.cpp` `#include`s directly. New test data must match the `package_data` globs in `setup.py` (and `MANIFEST.in`) to ship in the sdist.
- CI (`.github/workflows/test-python-package-with-conda.yml`) tests Python 3.8–3.12 against NumPy 1.19–2.0, so code must work on both NumPy 1.x and 2.x.

## Architecture

**Python layer** (`imread/imread.py`):
- Infers the format string from the file extension, or takes `formatstr` explicitly.
- Converts `os.PathLike` to `str`.
- Handles `as_grey` conversion in Python.
- Forwards everything else to `_imread`.
- `imread/special.py` maps formats that are handled outside C++ to Python readers. XCF, for example, shells out to `xcf2png`.
- `imread/ijrois.py` (ImageJ ROI reading) is pure Python.

**C++ core** (`imread/lib/`) does not depend on Python or NumPy, so it could be reused from other languages. Everything is built around a few abstractions in `lib/base.h`:
- `byte_source` / `byte_sink`: input and output streams.
  - `lib/file.h` provides `fd_source_sink` for file descriptors.
  - `lib/memory.h` provides `memory_source` for blobs.
- `Image` / `ImageFactory`: format code asks the factory to allocate an image, then fills it row by row through `rowp()`. It never touches NumPy directly.
- `ImageFormat`: one subclass per format, in `lib/_<fmt>.{h,cpp}`.
  - `can_read` / `can_read_multi` / `can_write` / `can_write_multi` / `can_write_metadata` advertise what the format supports.
  - `read` / `read_multi` / `write` / `write_multi` implement it.
- `options_map`: a `std::map<std::string, number_or_string>` built from the Python `opts` dict. Keys use `fmt:option` names (`jpeg:quality`, `tiff:compress`, `png:compression_level`). Read them with `get_optional_int`, `get_optional_bool` and `get_optional_cstring`.
- `lib/formats.cpp`:
  - `get_format()` maps a format string to an `ImageFormat`. Register new formats and aliases here (`jpg`→JPEG, `tif`→TIFF, `stk`→STK).
  - `magic_format()` handles magic-number detection. Only PNG and JPEG are detected today.

**NumPy glue**:
- `lib/numpy.{h,cpp}` implements `NumpyImage` and `NumpyFactory`. These are the `Image` and `ImageFactory` implementations backed by `PyArrayObject`.
- `imread/_imread.cpp` holds the CPython module functions: `imread`, `imread_multi`, `imread_from_blob`, `imsave`, `imsave_multi`, `detect_format`, `supports_format`.
  - It checks format capabilities and builds a `NumpyFactory`.
  - It translates C++ exceptions into Python exceptions. The exception types (`CannotReadError`, `CannotWriteError`, `NotImplementedError`, etc.) are defined in `lib/errors.h`.
  - `py_exception_set` signals that a Python error has already been set.

## Tests

- Tests live in `imread/tests/` (pytest).
- Use `from . import file_path` and `file_path('name')` to locate fixtures in `imread/tests/data/`.
- Malformed inputs for robustness tests live in `data/bad-files/`.
- Write tests save to a fixed filename in the current directory, such as `imread_testing_file.png`. An autouse fixture deletes that file.

## Conventions

- Commit messages start with a tag: `BUG`, `ENH`, `TST`, `DOC`, `BLD`, `RFCT`, `MIN`, `RLS`.
- User-visible changes go in `ChangeLog` and in the History section of `README.md`.
- The version string lives in `imread/imread_version.py`.
