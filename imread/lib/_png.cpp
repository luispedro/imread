// Copyright 2012-2016 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_png.h"
#include "tools.h"

#include <png.h>

#include <cstring>
#include <vector>
#include <sstream>

namespace {

void throw_error(png_structp png_ptr, png_const_charp error_msg) {
    throw CannotReadError(error_msg);
}

// This checks how 16-bit uints are stored in the current platform.
bool is_big_endian() {
    uint16_t v = 0xff00;
    unsigned char* vp = reinterpret_cast<unsigned char*>(&v);
    return (*vp == 0xff);
}

class png_holder {
    public:
        png_holder(int m)
            :png_ptr((m == write_mode ? png_create_write_struct : png_create_read_struct)(PNG_LIBPNG_VER_STRING, 0, throw_error, 0))
            ,png_info(0)
            ,mode(holder_mode(m))
            { }
        ~png_holder() {
            png_infopp pp = (png_info ? &png_info : 0);
            if (mode == read_mode) png_destroy_read_struct(&png_ptr, pp, 0);
            else png_destroy_write_struct(&png_ptr, pp);
        }
        void create_info() {
            png_info = png_create_info_struct(png_ptr);
            if (!png_info) throw ProgrammingError("Error in png_create_info_struct");
        }

        png_structp png_ptr;
        png_infop png_info;
        enum holder_mode { read_mode, write_mode } mode;
};

void read_from_source(png_structp png_ptr, png_byte* buffer, size_t n) {
    byte_source* s = static_cast<byte_source*>(png_get_io_ptr(png_ptr));
    const size_t actual = s->read(reinterpret_cast<byte*>(buffer), n);
    if (actual != n) {
        throw CannotReadError();
    }
}

void write_to_source(png_structp png_ptr, png_byte* buffer, size_t n) {
    byte_sink* s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
    const size_t actual = s->write(reinterpret_cast<byte*>(buffer), n);
    if (actual != n) {
        throw CannotReadError();
    }
}
void flush_source(png_structp png_ptr) {
    byte_sink* s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
    s->flush();
}

int color_type_of(Image* im) {
    if (im->nbits() != 8 && im->nbits() != 16) throw CannotWriteError("Image must be 8 or 16 bits for saving in PNG format");
    if (im->ndims() == 2) return PNG_COLOR_TYPE_GRAY;
    if (im->ndims() != 3) throw CannotWriteError("Image must be either 2 or 3 dimensional");
    if (im->dim(2) == 3) return PNG_COLOR_TYPE_RGB;
    if (im->dim(2) == 4) return PNG_COLOR_TYPE_RGBA;
    throw CannotWriteError();
}


void swap_bytes_inplace(std::vector<png_bytep>& data, const int nelems, stack_based_memory_pool& mem) {
    for (unsigned int r = 0; r != data.size(); ++r) {
        png_bytep row = data[r];
        png_bytep newbf = mem.allocate_as<png_bytep>(nelems * 2);
        std::memcpy(newbf, row, nelems*2);
        for (int c = 0; c != nelems; ++c) {
            std::swap(newbf[2*c], newbf[2*c + 1]);
        }
        data[r] = newbf;
    }
}
}

std::unique_ptr<Image> PNGFormat::read(byte_source* src, ImageFactory* factory, const options_map& opts) {
    png_holder p(png_holder::read_mode);
    png_set_read_fn(p.png_ptr, src, read_from_source);
    p.create_info();
    png_read_info(p.png_ptr, p.png_info);

    const int w = png_get_image_width (p.png_ptr, p.png_info);
    const int h = png_get_image_height(p.png_ptr, p.png_info);
    int bit_depth = png_get_bit_depth(p.png_ptr, p.png_info);
    if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
        std::ostringstream out;
        out << "imread.png: Cannot read this bit depth ("
                << bit_depth
                << "). Only bit depths ∈ {1,8,16} are supported.";
        throw CannotReadError(out.str());
    }

    // PNGs are in "network" order (ie., big-endian)
    if (bit_depth == 16 && !is_big_endian()) png_set_swap(p.png_ptr);

    const bool strip_alpha = get_optional_bool(opts, "strip_alpha", false);
    if (strip_alpha) {
        png_set_strip_alpha(p.png_ptr);
    }
    int d = -1;
    switch (png_get_color_type(p.png_ptr, p.png_info)) {
        case PNG_COLOR_TYPE_PALETTE:
            png_set_palette_to_rgb(p.png_ptr);
        case PNG_COLOR_TYPE_RGB:
            d = 3;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            d = 4 - int(strip_alpha);
            break;
        case PNG_COLOR_TYPE_GRAY:
            d = -1;
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            if (!strip_alpha) {
                throw CannotReadError("imread.png: Color type (4: grayscale with alpha channel) can  only be read when strip_alpha is set to true.");
            }
            d = -1;
            break;
        default: {
            std::ostringstream out;
            out << "imread.png: Color type ("
                << int(png_get_color_type(p.png_ptr, p.png_info))
                << ") cannot be handled";
            throw CannotReadError(out.str());
        }
    }

    std::unique_ptr<Image> output(factory->create(bit_depth, h, w, d));
    std::vector<png_bytep> rowps = allrows<png_byte>(*output);
    png_read_image(p.png_ptr, &rowps[0]);

    return output;
}

void PNGFormat::write(Image* input, byte_sink* output, const options_map& opts) {
    png_holder p(png_holder::write_mode);
    stack_based_memory_pool alloc;
    p.create_info();
    png_set_write_fn(p.png_ptr, output, write_to_source, flush_source);
    const int height = input->dim(0);
    const int width = input->dim(1);
    const int nchannels = input->ndims() == 2 ? 1 : input->dim(2);
    const int bit_depth = input->nbits();
    const int color_type = color_type_of(input);

    png_set_IHDR(p.png_ptr, p.png_info, width, height,
                     bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    int compression_level = get_optional_int(opts, "png:compression_level", -1);
    if (compression_level != -1) {
        png_set_compression_level(p.png_ptr, compression_level);
    }
    png_write_info(p.png_ptr, p.png_info);

    std::vector<png_bytep> rowps = allrows<png_byte>(*input);
    if (bit_depth == 16 && !is_big_endian()) {
        swap_bytes_inplace(rowps, width * nchannels, alloc);
    }

    png_write_image(p.png_ptr, &rowps[0]);
    png_write_end(p.png_ptr, p.png_info);
}
