// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
#define LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012

#include "base.h"

class TIFFFormat : public ImageFormat {
    public:
        bool can_read() const { return true; }
        bool can_read_multi() const { return true; }
        bool can_write() const { return true; }

        std::auto_ptr<Image> read(byte_source* s, ImageFactory* f) {
            std::auto_ptr<image_list> pages = this->do_read(s, f, false);
            if (pages->size() != 1) throw ProgrammingError();
            std::vector<Image*> ims = pages->release();
            return std::auto_ptr<Image>(ims[0]);
        }

        std::auto_ptr<image_list> read_multi(byte_source* s, ImageFactory* f) {
            return this->do_read(s, f, true);
        }
        void write(Image* input, byte_sink* output);
    private:
        std::auto_ptr<image_list> do_read(byte_source* s, ImageFactory* f, bool is_multi);
};


class STKFormat : public ImageFormat {
    public:
        bool can_read() const { return false; }
        bool can_read_multi() const { return true; }
        bool can_write() const { return false; }

        std::auto_ptr<image_list> read_multi(byte_source* s, ImageFactory* f);
};


#endif // LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
