// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <inttypes.h>
#include <unistd.h>
#include <memory>

#include "errors.h"

typedef uint8_t byte;

struct seekable {
    virtual ~seekable() { }
    virtual bool can_seek() const { return false; }

    virtual size_t seek_absolute(size_t) { throw NotImplementedError(); }
    virtual size_t seek_relative(int) { throw NotImplementedError(); }
    virtual size_t seek_end(int) { throw NotImplementedError(); }
};

class byte_source : virtual public seekable {
    public:
        virtual ~byte_source() { }
        virtual size_t read(byte* buffer, size_t) = 0;
};

class byte_sink : virtual public seekable {
    public:
        virtual ~byte_sink() { }

        virtual size_t write(const byte* buffer, size_t n) = 0;
        virtual void flush() { }
};

class Image {
    public:
        virtual ~Image() { }

        virtual void* rowp(int r) = 0;

        virtual int ndims() const = 0;
        virtual int dim(int) const = 0;

        virtual int dim_or(int dim, int def) const {
            if (dim >= this->ndims()) return def;
            return this->dim(dim);
        }

        template<typename T>
        T* rowp_as(const int r) {
            return static_cast<T*>(this->rowp(r));
        }
};

class ImageFactory {
    // This might seem a bit of over-engineering, but it is actually a very
    // clean interface.
    //
    // When calling the read method, it is not yet known what the type and the
    // dimensions of the image are going to have to be.
    //
    // The trick with the protected method makes the public interface a bit
    // cleaner and simulates virtual template functions.
    public:
        virtual ~ImageFactory() { }
        template <typename T>
        Image* create(int w, int h) { return this->do_create(code_for<T>(), w, h, -1); }

        template <typename T>
        Image* create(int w, int h, int d) { return this->do_create(code_for<T>(), w, h, d); }

    protected:
        enum type_code { uint8_v, uint16_v, uint32_v };
        virtual Image* do_create(type_code, int w, int h, int d) = 0;

        template <typename T>
        type_code code_for();
};

template <> inline ImageFactory::type_code ImageFactory::code_for<uint8_t>() { return uint8_v; }
template <> inline ImageFactory::type_code ImageFactory::code_for<uint16_t>() { return uint16_v; }
template <> inline ImageFactory::type_code ImageFactory::code_for<uint32_t>() { return uint32_v; }

class ImageFormat {
    public:
        virtual ~ImageFormat() { }

        virtual bool can_read() const { return false; }
        virtual bool can_write() const { return false; }

        virtual std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory) {
            throw NotImplementedError();
        }
        virtual void write(Image* input, byte_sink* output) {
            throw NotImplementedError();
        }
};

#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
