// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#include "base.h"
#include "tools.h"
#include "_bmp.h"
#include <sstream>

namespace {
void flippixels(byte* row, const int n) {
    for (int i = 0; i != n; row += 3, ++i) {
        byte b = row[0];
        byte g = row[1];
        byte r = row[2];
        row[0] = r;
        row[1] = g;
        row[2] = b;
    }
}
}

std::auto_ptr<Image> BMPFormat::read(byte_source* src, ImageFactory* factory) {
    char magick[2];
    if (src->read(reinterpret_cast<byte*>(magick), 2) != 2) {
        throw CannotReadError("imread.bmp: File is empty");
    }
    if (magick[0] != 'B' || magick[1] != 'M') {
        throw CannotReadError("imread.bmp: Magick number not matched (this might not be a BMP file)");
    }
    const uint32_t size = read32_le(*src);
    (void)read16_le(*src);
    (void)read16_le(*src);
    const uint32_t offset = read32_le(*src);
    const uint32_t hsize = read32_le(*src);
    const uint32_t width = read32_le(*src);
    const uint32_t height = read32_le(*src);
    const uint16_t planes = read16_le(*src);
    if (planes != 1){
        throw NotImplementedError("imread.bmp: planes should be 1");
    }
    const uint16_t bitsppixel = read16_le(*src);
    const uint32_t compression = read32_le(*src);
    if (compression != 0) {
        throw NotImplementedError("imread.bmp: Only uncompressed bitmaps are supported");
    }
    const uint32_t imsize = read32_le(*src);
    const uint32_t hres = read32_le(*src);
    const uint32_t vres = read32_le(*src);
    const uint32_t n_colours = read32_le(*src);
    const uint32_t importantcolours = read32_le(*src);

    if (bitsppixel != 8 && bitsppixel != 16 && bitsppixel != 24) {
        std::ostringstream out;
        out << "imread.bmp: Bits per pixel is " << bitsppixel << ". Only 8, 16, or 24 supported.";
        throw CannotReadError(out.str());
    }
    const int depth = (bitsppixel == 24 ? 3 : -1);
    const int nbits = (bitsppixel == 24? 8 : bitsppixel);
    std::auto_ptr<Image> output(factory->create(nbits, height, width, depth));
    const int bytes_per_row = width * (bitsppixel/8);
    const int padding = (4 - (bytes_per_row % 4)) % 4;
    byte buf[4];
    for (int r = 0; r != height; ++r) {
        byte* rowp = output->rowp_as<byte>(height-r-1);
        if (src->read(rowp, bytes_per_row) != bytes_per_row) {
            throw CannotReadError("File ended prematurely");
        }

        if (bitsppixel == 24) flippixels(rowp, width);

        (void)src->read(buf, padding);
    }
    return output;
}
