// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
#define LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
class TIFFFormat : public ImageFormat {
    public:
        bool can_read() const { return true; }
        bool can_write() const { return true; }

        std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory);
        void write(Image* input, byte_sink* output);
};


#endif // LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
