// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_tiff.h"
#include "tools.h"
#include <iostream>

extern "C" {
   #include <tiffio.h>
}

namespace {
tsize_t tiff_read(thandle_t handle, void* data, tsize_t n) {
    byte_source* s = static_cast<byte_source*>(handle);
    return s->read(static_cast<byte*>(data), n);
}
tsize_t tiff_write(thandle_t handle, void* data, tsize_t n) {
    byte_sink* s = static_cast<byte_sink*>(handle);
    return s->write(static_cast<byte*>(data), n);
}
template<typename T>
toff_t tiff_seek(thandle_t handle, toff_t off, int whence) {
    std::cerr << "SEEK(" << off << ", " << whence << ")\n";
    T* s = static_cast<T*>(handle);
    switch (whence) {
        case SEEK_SET: s->seek_absolute(off); return s->position();
        case SEEK_CUR: s->seek_relative(off); return s->position();
        case SEEK_END: s->seek_end(off); return s->position();
    }
    std::cerr << "still here\n";
    return -1;
}
int tiff_close(thandle_t handle) { return 0; }
toff_t tiff_size(thandle_t handle) {
    throw NotImplementedError();
}

struct tif_holder {
    tif_holder(TIFF* tif)
        :tif(tif) { }
    ~tif_holder() { TIFFClose(tif); }
    TIFF* tif;
};
} // namespace


void TIFFFormat::read(byte_source* src, Image* output) {
    tif_holder t = TIFFClientOpen(
                    "internal",
                    "r",
                    src,
                    tiff_read,
                    tiff_write,
                    tiff_seek<byte_source>,
                    tiff_close,
                    tiff_size,
                    NULL,
                    NULL);
    uint32 w, h;
	TIFFGetField(t.tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(t.tif, TIFFTAG_IMAGELENGTH, &h);
    output->set_size(h, w, 4);
    if (!TIFFReadRGBAImageOriented(t.tif, w, h, output->rowp_as<uint32>(0), ORIENTATION_TOPLEFT, 0)) {
        throw CannotReadError("Error reading TIFF file");
    }
}

void TIFFFormat::write(Image* input, byte_sink* output) {
    tif_holder t = TIFFClientOpen(
                    "internal",
                    "w",
                    output,
                    tiff_read,
                    tiff_write,
                    tiff_seek<byte_sink>,
                    tiff_close,
                    tiff_size,
                    NULL,
                    NULL);

    const uint32 h = input->dim(0);
    TIFFSetField(t.tif, TIFFTAG_IMAGELENGTH, uint32(h));
    TIFFSetField(t.tif, TIFFTAG_IMAGEWIDTH, uint32(input->dim(1)));

    TIFFSetField(t.tif, TIFFTAG_BITSPERSAMPLE, uint16(8));
    TIFFSetField(t.tif, TIFFTAG_SAMPLESPERPIXEL, uint16(input->dim_or(2, 1)));

    TIFFSetField(t.tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(t.tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

    for (uint32 r = 0; r != h; ++r) {
        if (TIFFWriteScanline(t.tif, input->rowp(r), r) == -1) {
            throw CannotWriteError("Error writing TIFF file");
        }
    }
    TIFFFlush(t.tif);
}
