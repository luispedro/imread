// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_pvrtc.h"
#include "tools.h"

#include "pvr.h"
#include <cstring>

std::auto_ptr<Image> PVRTCFormat::read(byte_source* src, ImageFactory* factory) {
    std::vector<byte> data = full_data(*src);
    PVRTexture pvr;

    bool res = pvr.loadApplePVRTC(&data[0], data.size());
    if (!res) {
        throw CannotReadError("imread.imread._pvrtc: File isn't a valid PVRTC texture.");
    }

    std::auto_ptr<Image> output(factory->create(8, pvr.height, pvr.width, 4));
    if (pvr.data) {
        /* clearly this is not the best approach */
        uint8_t* rowp = output->rowp_as<uint8_t>(0);
        std::memcpy(rowp, pvr.data, pvr.width*pvr.height*4);
    } else {
        throw CannotReadError("imread.imread._pvrtc: Error reading PVRTC file.");
    }

    return output;
}

