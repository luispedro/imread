// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include "base.h"
#include "_tiff.h"
#include "tools.h"

#include <sstream>

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
    T* s = static_cast<T*>(handle);
    switch (whence) {
        case SEEK_SET: return s->seek_absolute(off);
        case SEEK_CUR: return s->seek_relative(off);
        case SEEK_END: return s->seek_end(off);
    }
    return -1;
}
int tiff_close(thandle_t handle) { return 0; }
template<typename T>
toff_t tiff_size(thandle_t handle) {
    T* s = static_cast<T*>(handle);
    const size_t curpos = s->seek_relative(0);
    const size_t size = s->seek_end(0);
    s->seek_absolute(curpos);
    return size;
}

void tiff_error(const char* module, const char* fmt, va_list ap) {
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), fmt, ap);
    std::string error_message(buffer);
    throw CannotReadError(std::string("imread.imread._tiff: libtiff error: `") + buffer + std::string("`"));
}

struct tif_holder {
    tif_holder(TIFF* tif)
        :tif(tif)
        { TIFFSetErrorHandler(tiff_error); }
    ~tif_holder() { TIFFClose(tif); }
    TIFF* tif;
};

template <typename T>
inline
T tiff_get(const tif_holder& t, const int tag) {
    T val;
    if (!TIFFGetField(t.tif, tag, &val)) {
        std::stringstream out;
        out << "imread.imread._tiff: Cannot find necessary tag (" << tag << ")";
        throw CannotReadError(out.str());
    }
    return val;
}
template <typename T>
inline
T tiff_get(const tif_holder& t, const int tag, const T def) {
    T val;
    if (!TIFFGetField(t.tif, tag, &val)) return def;
    return val;
}
TIFF* read_client(byte_source* src) {
    return TIFFClientOpen(
                    "internal",
                    "r",
                    src,
                    tiff_read,
                    tiff_write,
                    tiff_seek<byte_source>,
                    tiff_close,
                    tiff_size<byte_source>,
                    NULL,
                    NULL);
}

const int UIC1Tag = 33628;
const int UIC2Tag = 33629;
const int UIC3Tag = 33630;
const int UIC4Tag = 33631;

const TIFFFieldInfo stkTags[] = {
    { UIC1Tag, -1,-1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
    { UIC1Tag, -1,-1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
    //{ UIC2Tag, -1, -1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
    { UIC2Tag, -1, -1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
    { UIC3Tag, -1,-1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC3Tag") },
    { UIC4Tag, -1,-1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC4Tag") },
};

void set_stk_tags(TIFF* tif) {
    TIFFMergeFieldInfo(tif, stkTags, sizeof(stkTags)/sizeof(stkTags[0]));
}



class shift_source : public byte_source {
    public:
        explicit shift_source(byte_source* s)
            :s(s)
            ,shift_(0)
            { }

        virtual size_t read(byte* buf, size_t n) { return s->read(buf, n); }

        virtual size_t seek_absolute(size_t pos) { return s->seek_absolute(pos + shift_)-shift_; }
        virtual size_t seek_relative(int n) { return s->seek_relative(n)-shift_; }
        virtual size_t seek_end(int n) { return s->seek_end(n+shift_)-shift_; }

        void shift(int nshift) {
            s->seek_relative(nshift - shift_);
            shift_ = nshift;
        }

        byte_source* s;
        int shift_;
};

struct stk_extend {
    stk_extend()
        :proc(TIFFSetTagExtender(set_stk_tags)) { }
    ~stk_extend() {
        TIFFSetTagExtender(proc);
    }
    TIFFExtendProc proc;
};

} // namespace


std::auto_ptr<image_list> STKFormat::read_multi(byte_source* src, ImageFactory* factory) {
    shift_source moved(src);
    stk_extend ext;
    tif_holder t = read_client(&moved);
    std::auto_ptr<image_list> images(new image_list);
    const uint32 h = tiff_get<uint32>(t, TIFFTAG_IMAGELENGTH);
    const uint32 w = tiff_get<uint32>(t, TIFFTAG_IMAGEWIDTH);

    const uint16 nr_samples = tiff_get<uint16>(t, TIFFTAG_SAMPLESPERPIXEL, 1);
    const uint16 bits_per_sample = tiff_get<uint16>(t, TIFFTAG_BITSPERSAMPLE, 8);
    const int depth = nr_samples > 1 ? nr_samples : -1;

    const int strip_size = TIFFStripSize(t.tif);
    const int n_strips = TIFFNumberOfStrips(t.tif);
    int32_t n_planes;
    void* data;
    TIFFGetField(t.tif, UIC3Tag, &n_planes, &data);
    int raw_strip_size = 0;
    for (int st = 0; st != n_strips; ++st) {
        raw_strip_size += TIFFRawStripSize(t.tif, st);
    }
    for (int z = 0; z < n_planes; ++z) {
        // Monkey patch strip offsets. This is very hacky, but it seems to work!
        moved.shift(z * raw_strip_size);

        std::auto_ptr<Image> output(factory->create(bits_per_sample, h, w, depth));
        uint8_t* start = output->rowp_as<uint8_t>(0);
        for (int st = 0; st != n_strips; ++st) {
            const int offset = TIFFReadEncodedStrip(t.tif, st, start, strip_size);
            if (offset == -1) {
                throw CannotReadError("imread.imread._tiff.stk: Error reading strip");
            }
            start += offset;
        }
        images->push_back(output);
    }
    return images;
}

std::auto_ptr<image_list> TIFFFormat::do_read(byte_source* src, ImageFactory* factory, bool is_multi) {
    tif_holder t = read_client(src);
    std::auto_ptr<image_list> images(new image_list);
    do {
        const uint32 h = tiff_get<uint32>(t, TIFFTAG_IMAGELENGTH);
        const uint32 w = tiff_get<uint32>(t, TIFFTAG_IMAGEWIDTH);
        const uint16 nr_samples = tiff_get<uint16>(t, TIFFTAG_SAMPLESPERPIXEL);
        const uint16 bits_per_sample = tiff_get<uint16>(t, TIFFTAG_BITSPERSAMPLE);
        const int depth = nr_samples > 1 ? nr_samples : -1;

        std::auto_ptr<Image> output(factory->create(bits_per_sample, h, w, depth));

        for (uint32 r = 0; r != h; ++r) {
            if(TIFFReadScanline(t.tif, output->rowp_as<byte>(r), r) == -1) {
                throw CannotReadError("imread.imread._tiff: Error reading scanline");
            }
        }
        images->push_back(output);
    } while (is_multi && TIFFReadDirectory(t.tif));
    return images;
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
                    tiff_size<byte_sink>,
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
            throw CannotWriteError("imread.imsave._tiff: Error writing TIFF file");
        }
    }
    TIFFFlush(t.tif);
}
