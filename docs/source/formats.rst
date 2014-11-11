=================
Formats Supported
=================

+---------------+------+-------+---------------+------------------------------+
| Format        | Read | Write | Since version |            Notes             |
+---------------+------+-------+---------------+------------------------------+
| PNG           |  Y   |   Y   |      0.1      |                              |
+---------------+------+-------+---------------+------------------------------+
| PNG (16 bits) |  Y   |   Y   |      0.4      |                              |
+---------------+------+-------+---------------+------------------------------+
| TIFF          |  Y   |   Y   |      0.1      |                              |
+---------------+------+-------+---------------+------------------------------+
| JPEG          |  Y   |   Y   |      0.1      |                              |
+---------------+------+-------+---------------+------------------------------+
| WEBP          |  Y   |   N   |      0.2      |                              |
+---------------+------+-------+---------------+------------------------------+
| BMP           |  Y   |   N   |      0.2.5    | Only uncompressed bitmaps    |
|               |      |       |               | are supported.               |
+---------------+------+-------+---------------+------------------------------+
| STK           |  Y   |   N   |      0.2      |                              |
+---------------+------+-------+---------------+------------------------------+
| LSM           |  Y   |   N   |      0.2.2    |                              |
+---------------+------+-------+---------------+------------------------------+
| XCF           |  Y   |   N   |      0.2.2    | Only if ``xcf2png`` utility  |
|               |      |       |               | is found in the path.        |
+---------------+------+-------+---------------+------------------------------+

Options
-------

Some of the formats allow you to specify options when saving. These are
inevitably format specific.

PNG
~~~

png:compression_level
    Compression level to use, from 0 (no compression) to 9. Setting it to 0 is discouraged.

JPEG
~~~~
jpeg:quality
    An integer 1-100 determining the quality used by JPEG backend
    (default is libjpeg default: 75).

TIFF
~~~~

tiff:compress
    Whether to use compression when saving TIFF (default: True)

tiff:horizontal-predictor
    Whether to use horizontal prediction in TIFF. This defaults to True
    for 16 bit images, and to False for 8 bit images. This is because
    compressing 16 bit images without horizontal prediction is often
    counter-productive (see http://www.asmail.be/msg0055176395.html)

