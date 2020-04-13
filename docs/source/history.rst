=======
History
=======

Version 0.7.3 (2020-04-09)
--------------------------
- Add missing header files to distribution

Version 0.7.3 (2020-04-09)
--------------------------
- Add missing test data to distribution

Version 0.7.2 (2020-03-24)
--------------------------
- Fix several memory access bugs in parsers (reported by Robert Scott)

Version 0.7.1 (2019-05-09)
--------------------------
- Fix 16-bit RGB/RGBA TIFF write (patch by Tomi Aarnio)

Version 0.7.0 (2018-09-30)
--------------------------
- Add support for reading ImageJ ROIs

Version 0.6.1 (2018-02-15)
--------------------------
- Support pathlib paths as function arguments
- Fix 16 bit PNG write support (patch by Tomi Aarnio)


Version 0.6 (2016-09-21)
--------------------------
- Add `supports_format` function
- Make png compression level tunable when calling imsave
- Add imsave_multi
- Add partial support for reading PNG files in Gray+alpha format


Version 0.5.1 (2014-11-06)
--------------------------
- Improve tests to work after installation
- Fix compilation in MSVC (patch by Christoph Gohlke)


Version 0.5 (2014-10-16)
------------------------
- Add magic-number based format auto-detection
- Auto detect whether webp is installed
- Fix WebP reading (update to newer API)


Version 0.4 (2014-07-21)
------------------------
- Add configuration for TIFF saving
- Correctly save 16 bit PNG images
- Better error messages for JPEG

Version 0.3.2 (2013-10-06)
--------------------------
- Added imload*/imwrite synonyms as suggested by Thouis (Ray) Jones
- Options framework
- Allow user to specify JPEG quality when saving
- Fix loading of 16 bit PNG images

Version 0.3.1 (2013-06-20)
--------------------------
- Fix possible crash on error with TIFF
- Fix compilation on Windows (reported by Volker Hilsenstein)
- Make it easy to compile without WebP

Version 0.3.0 (2013-07-29)
--------------------------
- Support for reading from in-memory blobs
- Support for reading & writing TIFF metadata
- Add PHOTOMETRIC tag to TIFF (reported by Volker Hilsenstein)
- Support writing RGB TIFFs

Version 0.2.6 (2013-06-19)
--------------------------
- Fix hard crash when saving with non-existing file type
- Fix compilation on MacOS (patch by Alexander Bohn)
- Add ``verbose`` argument to tests.run()
- Better error when attempting to save floating point images

Version 0.2.5 (2012-10-29)
--------------------------
- Correctly accept uppercase extensions
- Python 3 support (patch by Christoph Gohlke [pull request 8 on github])
- Read 1-Bit PNGs
- Read simple BMPs (compression and many bit types not supported)
- More complete debug mode (export DEBUG=2 when building), more checks

Version 0.2.4 (2012-06-26)
--------------------------
- Add lzw.cpp to source distribution
- Support saving 16-bit TIFF
- Better Mac OS support (patch from Alexander Bohn)

Version 0.2.3 (2012-06-8)
-------------------------
- Fix imread_multi

Version 0.2.2 (2012-06-5)
-------------------------
- Add `formatstr` argument to imread
- Open files in binary mode on Windows (patch by Christoph Gohlke)
- Read-only support for LSM files
- Read-only support for XCF files (through `xcf2png`)
- Fix writing of non-contiguous images (at least PNG was affected)


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

