// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "formats.h"
#include "_png.h"
#include "_jpeg.h"
#include "_tiff.h"

#include <cstring>

std::auto_ptr<ImageFormat> get_format(const char* format) {
    using std::strcmp;
    if (!strcmp(format, "png")) return std::auto_ptr<ImageFormat>(new PNGFormat);
    if (!strcmp(format, "jpeg") || !strcmp(format, "jpg")) return std::auto_ptr<ImageFormat>(new JPEGFormat);
    if (!strcmp(format, "tiff") || !strcmp(format, "tif")) return std::auto_ptr<ImageFormat>(new TIFFFormat);
    return std::auto_ptr<ImageFormat>(0);
}
