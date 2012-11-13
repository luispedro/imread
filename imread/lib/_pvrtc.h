// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PVRTC_H_INCLUDE_GUARD_
#define LPC_PVRTC_H_INCLUDE_GUARD_

#include "base.h"

class PVRTCFormat : public ImageFormat {
    public:
        bool can_read() const { return true; }
        bool can_write() const { return false; }

        std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory);
};

#endif // LPC_PVRTC_H_INCLUDE_GUARD_
