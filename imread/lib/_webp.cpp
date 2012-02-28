// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_webp.h"
#include "tools.h"

#include <webp/decode.h>

std::auto_ptr<Image> WebPFormat::read(byte_source* src, ImageFactory* factory) {
    std::vector<byte> data = full_data(*src);
    int w, h;
    int err = WebPGetInfo(&data[0], data.size(), &w, &h);
    if (err) {
        throw CannotReadError();
    }
    std::auto_ptr<Image> output(factory->create<byte>(h, w, 4));
    const int stride = w*4;
    const uint8_t* errp = WebPDecodeRGBAInto(
            &data[0], data.size(),
            output->rowp_as<byte>(0), h*stride, stride);
    if (errp) {
        throw CannotReadError();
    }

    return output;
}

