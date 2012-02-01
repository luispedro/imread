#include "formats.h"
#include "_png.h"
#include <cstring>

std::auto_ptr<ImageFormat> get_format(const char* format) {
    if (!std::strcmp(format, "png")) return std::auto_ptr<ImageFormat>(new PNGFormat);
    return std::auto_ptr<ImageFormat>(0);
}
