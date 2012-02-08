// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_tiff.h"
#include "tools.h"

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
toff_t tiff_src_seek(thandle_t handle, toff_t off, int whence) {
    byte_source* s = static_cast<byte_source*>(handle);
    switch (whence) {
        case SEEK_SET: s->seek_absolute(off); return s->position();
        case SEEK_CUR: s->seek_relative(off); return s->position();
    }
    return -1;
}
int tiff_close(thandle_t handle) { return 0; }
toff_t tiff_size(thandle_t handle) {
    throw NotImplementedError();
}

} // namespace


void TIFFFormat::read(byte_source* src, Image* output) {
    TIFF* tif = TIFFClientOpen(
                    "internal",
                    "r",
                    src,
                    tiff_read,
                    tiff_write,
                    tiff_src_seek,
                    tiff_close,
                    tiff_size,
                    NULL,
                    NULL);
    uint32 w, h;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    output->set_size(h, w, 4);
    const int ret = TIFFReadRGBAImageOriented(tif, w, h, output->rowp_as<uint32>(0), ORIENTATION_TOPLEFT, 0);
    TIFFClose(tif);
    if (!ret) throw CannotReadError("Error reading TIFF file");
}
