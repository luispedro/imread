// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_webp.h"
#include "tools.h"

#include <webp/decode.h>

std::auto_ptr<Image> WebPFormat::read(byte_source* src, ImageFactory* factory) {
    std::vector<byte> data = full_data(*src);
    int w, h;
    int ok = WebPGetInfo(&data[0], data.size(), &w, &h);
    if (!ok) {
        throw CannotReadError("imread.imread._webp: File does not validate as WebP");
    }
    std::auto_ptr<Image> output(factory->create(8, h, w, 4));
    const int stride = w*4;
    const uint8_t* p = WebPDecodeRGBAInto(
            &data[0], data.size(),
            output->rowp_as<byte>(0), h*stride, stride);
    if (p != output->rowp_as<uint8_t>(0)) {
        throw CannotReadError("imread.imread._webp: Error in decoding file");
    }

    return output;
}

