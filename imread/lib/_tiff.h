// Copyright 2012-2019 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
#define LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012

#include "base.h"

class TIFFFormat : public ImageFormat {
    public:
        bool can_read()             const { return true; }
        bool can_read_multi()       const { return true; }
        bool can_write()            const { return true; }
        bool can_write_multi()      const { return true; }
        bool can_write_metadata()   const { return true; }

        std::unique_ptr<Image> read(byte_source* s, ImageFactory* f, const options_map& opts) {
            std::unique_ptr<image_list> pages = this->do_read(s, f, false);
            if (pages->size() != 1) throw ProgrammingError();
            std::vector<Image*> ims = pages->release();
            return std::unique_ptr<Image>(ims[0]);
        }

        std::unique_ptr<image_list> read_multi(byte_source* s, ImageFactory* f, const options_map& opts) {
            return this->do_read(s, f, true);
        }
        void write(Image* input, byte_sink* output, const options_map& opts);
        void write_multi(image_list* input, byte_sink* output, const options_map& opts);
    private:
        std::unique_ptr<image_list> do_read(byte_source* s, ImageFactory* f, bool is_multi);
        void do_write(image_list* input, byte_sink* output, const options_map& opts, bool);
};


class STKFormat : public ImageFormat {
    public:
        bool can_read() const { return false; }
        bool can_read_multi() const { return true; }
        bool can_write() const { return false; }

        std::unique_ptr<image_list> read_multi(byte_source* s, ImageFactory* f);
};


#endif // LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
