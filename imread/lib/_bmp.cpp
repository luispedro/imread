// Copyright 2012-2020 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#define NO_IMPORT_ARRAY
#include "base.h"
#include "tools.h"
#include "_bmp.h"
#include <sstream>

namespace {
const uint32_t BI_RGB = 0;
const uint32_t BI_BITFIELDS = 3;
const uint32_t BI_ALPHABITFIELDS = 6;

const uint32_t BITMAPINFOHEADER_SIZE = 40;
const uint32_t BITMAPV2INFOHEADER_SIZE = 52;
const uint32_t BITMAPV3INFOHEADER_SIZE = 56;

// Extracts one channel described by a bit mask (as used in BI_BITFIELDS
// images) and scales it to 8 bits
struct channel_mask {
    channel_mask()
        :mask_(0)
        ,shift_(0)
        ,bits_(0)
        { }
    explicit channel_mask(const uint32_t mask)
        :mask_(mask)
        ,shift_(0)
        ,bits_(0) {
        if (!mask) return;
        while (!((mask >> shift_) & 1)) ++shift_;
        const uint32_t m = mask >> shift_;
        if (m & (m + 1)) {
            throw CannotReadError("imread.bmp: Malformed BMP file: colour mask is not contiguous");
        }
        while (bits_ < 32 && ((m >> bits_) & 1)) ++bits_;
    }

    byte extract(const uint32_t pixel) const {
        if (!bits_) return 0;
        const uint32_t v = (pixel & mask_) >> shift_;
        if (bits_ >= 8) return byte(v >> (bits_ - 8));
        const uint32_t maxv = (uint32_t(1) << bits_) - 1;
        return byte((v * 255 + maxv/2) / maxv);
    }
    bool present() const { return mask_ != 0; }

    private:
        uint32_t mask_;
        int shift_;
        int bits_;
};

uint32_t read_pixel_le(const byte* p, const int bytes) {
    uint32_t v = 0;
    for (int b = bytes - 1; b >= 0; --b) v = (v << 8) | p[b];
    return v;
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
    if (header_size < BITMAPINFOHEADER_SIZE) {
        std::ostringstream out;
        out << "imread.bmp: Header size is " << header_size << ". Only BITMAPINFOHEADER (40 bytes) or later is supported.";
        throw NotImplementedError(out.str());
    }
    const int32_t width = int32_t(read32_le(*src));
    // A negative height signals a top-down bitmap
    const int32_t signed_height = int32_t(read32_le(*src));
    const uint16_t planes = read16_le(*src);
    if (planes != 1){
        throw NotImplementedError("imread.bmp: planes should be 1");
    }
    const uint16_t bitsppixel = read16_le(*src);
    const uint32_t compression = read32_le(*src);
    const uint32_t imsize = read32_le(*src);
    (void)imsize;
    const uint32_t hres = read32_le(*src);
    (void)hres;
    const uint32_t vres = read32_le(*src);
    (void)vres;
    const uint32_t n_colours = read32_le(*src);
    const uint32_t importantcolours = read32_le(*src);
    (void)importantcolours;

    if (bitsppixel != 1 && bitsppixel != 4 && bitsppixel != 8 &&
            bitsppixel != 16 && bitsppixel != 24 && bitsppixel != 32) {
        std::ostringstream out;
        out << "imread.bmp: Bits per pixel is " << bitsppixel << ". Only 1, 4, 8, 16, 24, or 32 supported.";
        throw CannotReadError(out.str());
    }
    const bool uses_masks = (bitsppixel == 16 || bitsppixel == 32);
    if (compression != BI_RGB &&
            !(uses_masks && (compression == BI_BITFIELDS || compression == BI_ALPHABITFIELDS))) {
        throw NotImplementedError("imread.bmp: Only uncompressed bitmaps are supported");
    }
    if (width <= 0 || signed_height == 0 || signed_height == INT32_MIN) {
        throw CannotReadError("imread.bmp: Malformed BMP file: invalid image dimensions");
    }
    const bool top_down = (signed_height < 0);
    const int height = (top_down ? -signed_height : signed_height);

    channel_mask red, green, blue, alpha;
    if (compression == BI_RGB) {
        if (bitsppixel == 16) {
            red = channel_mask(0x7c00);
            green = channel_mask(0x03e0);
            blue = channel_mask(0x001f);
        } else if (bitsppixel == 32) {
            // The high byte is unused
            red = channel_mask(0x00ff0000);
            green = channel_mask(0x0000ff00);
            blue = channel_mask(0x000000ff);
        }
    } else {
        // In a BITMAPINFOHEADER, the masks follow the header. In later
        // versions, they are part of the header.
        // Both layouts place them right after the first 40 bytes.
        const uint32_t red_mask = read32_le(*src);
        const uint32_t green_mask = read32_le(*src);
        const uint32_t blue_mask = read32_le(*src);
        const bool has_alpha_mask = (compression == BI_ALPHABITFIELDS || header_size >= BITMAPV3INFOHEADER_SIZE);
        const uint32_t alpha_mask = (has_alpha_mask ? read32_le(*src) : 0);
        if (header_size > BITMAPINFOHEADER_SIZE && header_size < BITMAPV2INFOHEADER_SIZE) {
            throw CannotReadError("imread.bmp: Malformed BMP file: header is too small to hold colour masks");
        }
        red = channel_mask(red_mask);
        green = channel_mask(green_mask);
        blue = channel_mask(blue_mask);
        alpha = channel_mask(alpha_mask);
    }

    std::vector<byte> color_table;
    if (bitsppixel <= 8) {
        // Larger tables are legal, but entries past 2**bitsppixel cannot be used
        const uint32_t max_colours = (uint32_t(1) << bitsppixel);
        const uint32_t table_size = ((n_colours == 0 || n_colours > max_colours) ? max_colours : n_colours);
        color_table.resize(table_size*4);
        src->seek_absolute(14 + size_t(header_size));
        src->read_check(color_table.data(), color_table.size());
    }

    const int depth = (alpha.present() ? 4 : 3);
    std::unique_ptr<Image> output(factory->create(8, height, width, depth));

    // Rows are padded to a multiple of 4 bytes
    const size_t bytes_per_row = (size_t(width) * bitsppixel + 7) / 8;
    const size_t stride = (size_t(width) * bitsppixel + 31) / 32 * 4;
    std::vector<byte> buf(stride);

    src->seek_absolute(offset);
    for (int r = 0; r != height; ++r) {
        // Some writers omit the padding of the last row
        const size_t n = src->read(buf.data(), stride);
        if (n < bytes_per_row || (n != stride && r != (height - 1))) {
            throw CannotReadError("imread.bmp: File ended prematurely");
        }
        byte* rowp = output->rowp_as<byte>(top_down ? r : height - r - 1);
        const byte* in = buf.data();

        if (bitsppixel == 24) {
            for (int i = 0; i != width; ++i, in += 3, rowp += 3) {
                rowp[0] = in[2];
                rowp[1] = in[1];
                rowp[2] = in[0];
            }
        } else if (uses_masks) {
            const int bytes = bitsppixel/8;
            for (int i = 0; i != width; ++i, in += bytes, rowp += depth) {
                const uint32_t pixel = read_pixel_le(in, bytes);
                rowp[0] = red.extract(pixel);
                rowp[1] = green.extract(pixel);
                rowp[2] = blue.extract(pixel);
                if (depth == 4) rowp[3] = alpha.extract(pixel);
            }
        } else {
            const int index_mask = (1 << bitsppixel) - 1;
            for (int i = 0; i != width; ++i, rowp += 3) {
                const size_t bit = size_t(i) * bitsppixel;
                const int shift = 8 - bitsppixel - int(bit % 8);
                const size_t index = (in[bit / 8] >> shift) & index_mask;
                if (4*index + 3 > color_table.size()) {
                    throw CannotReadError("imread.bmp: Malformed BMP file: color table is too small");
                }
                // The colour table is stored as BGRX
                rowp[0] = color_table[4*index + 2];
                rowp[1] = color_table[4*index + 1];
                rowp[2] = color_table[4*index + 0];
            }
        }
    }
    return output;
}
