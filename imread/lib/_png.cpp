// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_png.h"

#include <png.h>

#include <cstring>
#include <vector>

class png_holder {
    public:
        png_holder()
            :png_ptr(png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0))
            ,png_info(0)
            { }
        ~png_holder() {
            if (png_info) png_destroy_read_struct(&png_ptr, &png_info, 0);
            else png_destroy_read_struct(&png_ptr, 0, 0);
        }
        void create_info() {
            png_info = png_create_info_struct(png_ptr);
            if (!png_info) throw "error";
        }

        png_structp png_ptr;
        png_infop png_info;
};

void read_from_source(png_structp png_ptr, png_byte* buffer, size_t n) {
    byte_source* s = static_cast<byte_source*>(png_get_io_ptr(png_ptr));
    const size_t actual = s->read(reinterpret_cast<byte*>(buffer), n);
    if (actual != n) {
        throw CannotReadError();
    }
}

void PNGFormat::read(byte_source* src, Image* output) {
    png_holder p;
    if (setjmp(png_jmpbuf(p.png_ptr))) {
        throw CannotReadError();
    }
    png_set_read_fn(p.png_ptr, src, read_from_source);
    p.create_info();
    png_read_info(p.png_ptr, p.png_info);
    
    const int w = png_get_image_width (p.png_ptr, p.png_info);
    const int h = png_get_image_height(p.png_ptr, p.png_info);
    int d = -1;
    switch (png_get_color_type(p.png_ptr, p.png_info)) {
        case PNG_COLOR_TYPE_RGB:
            d = 3;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            d = 4;
            break;
        case PNG_COLOR_TYPE_GRAY:
            d = -1;
            break
        default:
            throw CannotReadError("Unhandled color type");
    }

    output->set_size(h, w, d);
    std::vector<png_bytep> rowps;
    for (int r = 0; r != h; ++r) {
        png_byte* rowp = output->rowp_as<png_byte>(r);
        rowps.push_back(rowp);
    }
    png_read_image(p.png_ptr, &rowps[0]);
}


