// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_pvrtc.h"
#include "tools.h"

#include "pvr.h"

std::auto_ptr<Image> WebPFormat::read(byte_source* src, ImageFactory* factory) {
    std::vector<byte> data = full_data(*src);
    PVRTexture pvr;

    int res = pvr.load(str(data.begin(), data.end()));
    if (res != PVR_LOAD_OKAY && res != PVR_LOAD_UNKNOWN_TYPE) {
        throw CannotReadError("imread.imread._pvrtc: File isn't a valid PVRTC texture.");
    }

    std::auto_ptr<Image> output(factory->create(8, pvr.height, pvr.width, 4));
    if (pvr.data) {
        output->rowp_as<uint8_t>(0) = pvr.data
    } else {
        throw CannotReadError("imread.imread._pvrtc: Error reading PVRTC file.");
    }

    return output;
}

