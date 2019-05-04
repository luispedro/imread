// Copyright 2012-2019 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "formats.h"
#include "_bmp.h"
#include "_jpeg.h"
#include "_lsm.h"
#include "_png.h"
#include "_tiff.h"

#ifndef IMREAD_EXCLUDE_WEBP
#include "_webp.h"
#endif

#include <cstring>

std::unique_ptr<ImageFormat> get_format(const char* format) {
    using std::strcmp;
    if (!strcmp(format, "png")) return std::unique_ptr<ImageFormat>(new PNGFormat);
    if (!strcmp(format, "jpeg") || !strcmp(format, "jpg")) return std::unique_ptr<ImageFormat>(new JPEGFormat);
    if (!strcmp(format, "lsm")) return std::unique_ptr<ImageFormat>(new LSMFormat);
    if (!strcmp(format, "tiff") || !strcmp(format, "tif")) return std::unique_ptr<ImageFormat>(new TIFFFormat);

#if IMREAD_EXCLUDE_WEBP
    if (!strcmp(format, "webp")) return std::unique_ptr<ImageFormat>(nullptr);
#else
    if (!strcmp(format, "webp")) return std::unique_ptr<ImageFormat>(new WebPFormat);
#endif

    if (!strcmp(format, "stk")) return std::unique_ptr<ImageFormat>(new STKFormat);
    if (!strcmp(format, "bmp")) return std::unique_ptr<ImageFormat>(new BMPFormat);
    return std::unique_ptr<ImageFormat>(nullptr);
}

const char* magic_format(byte_source* src) {
    if (PNGFormat::match_format(src)) return "png";
    if (JPEGFormat::match_format(src)) return "jpeg";
    return 0;
}
