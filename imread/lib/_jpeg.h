// Copyright 2012-2019 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
#define LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012

#include "base.h"

class JPEGFormat : public ImageFormat {
    public:
        bool can_read() const { return true; }
        bool can_write() const { return true; }

        static bool match_format(byte_source* src) { return match_magic(src, "\xff\xd8\xff", 3); }

        std::unique_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map& opts);
        void write(Image* input, byte_sink* output, const options_map& opts);
};


#endif // LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
