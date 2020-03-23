// Copyright 2012-2020 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#define NO_IMPORT_ARRAY
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

void color_expand(const std::vector<byte>& color_table, byte* row, const int w) {
    // We are expanding in-place
    // This means that we must process the row backwards, which is slightly
    // awkward, but works correctly
    std::vector<byte>::const_iterator pbegin = color_table.begin();
    for (int i = w-1; i >= 0; --i) {
        if (color_table.size() < 4*row[i] + 3) {
            throw CannotReadError("Malformed BMP file: color table is too small");
        }
        std::copy(pbegin + 4*row[i], pbegin + 4*row[i] + 3, row + 3*i);
    }
}

uint32_t pow2(uint32_t n) {
    if (n <= 0) return 1;
    return 2*pow2(n-1);
}
}

std::unique_ptr<Image> BMPFormat::read(byte_source* src, ImageFactory* factory, const options_map&) {
    char magick[2];
    if (src->read(reinterpret_cast<byte*>(magick), 2) != 2) {
        throw CannotReadError("imread.bmp: File is empty");
    }
    if (magick[0] != 'B' || magick[1] != 'M') {
        throw CannotReadError("imread.bmp: Magick number not matched (this might not be a BMP file)");
    }
    const uint32_t size = read32_le(*src);
    (void)size;
    (void)read16_le(*src);
    (void)read16_le(*src);
    const uint32_t offset = read32_le(*src);
    const uint32_t header_size = read32_le(*src);
    (void)header_size;
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
    (void)imsize;
    const uint32_t hres = read32_le(*src);
    (void)hres;
    const uint32_t vres = read32_le(*src);
    (void)vres;
    const uint32_t n_colours = read32_le(*src);
    const uint32_t importantcolours = read32_le(*src);
    (void)importantcolours;

    if (bitsppixel != 8 && bitsppixel != 16 && bitsppixel != 24) {
        std::ostringstream out;
        out << "imread.bmp: Bits per pixel is " << bitsppixel << ". Only 8, 16, or 24 supported.";
        throw CannotReadError(out.str());
    }
    const int depth = (bitsppixel == 16 ? -1 : 3);
    const int nbits = (bitsppixel == 16 ? 16 : 8);
    std::unique_ptr<Image> output(factory->create(nbits, height, width, depth));

    std::vector<byte> color_table;
    if (bitsppixel <= 8) {
        const uint32_t table_size = (n_colours == 0 ? pow2(bitsppixel) : n_colours);
        color_table.resize(table_size*4);
        src->read_check(color_table.data(), table_size*4);
    }

    src->seek_absolute(offset);
    const int bytes_per_row = width * (bitsppixel/8);
    const int padding = (4 - (bytes_per_row % 4)) % 4;
    byte buf[4];
    for (unsigned int r = 0; r != height; ++r) {
        byte* rowp = output->rowp_as<byte>(height-r-1);
        src->read_check(rowp, bytes_per_row);

        if (bitsppixel == 24) flippixels(rowp, width);
        else if (!color_table.empty()) color_expand(color_table, rowp, width);

        if (src->read(buf, padding) != unsigned(padding) && r != (height - 1)) {
            throw CannotReadError("File ended prematurely");
        }
    }
    return output;
}

