#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <inttypes.h>
#include <unistd.h>

#include "errors.h"

typedef uint8_t byte;

class byte_source {
    public:
        virtual ~byte_source() { }

        virtual size_t read(byte* buffer, size_t) = 0;

        virtual bool can_seek() const { return false; }
        virtual void seek_absolute(size_t) { throw NotImplementedError(); }
        virtual void seek_relative(int) { throw NotImplementedError(); }
};

class byte_sink {
    public:
        virtual ~byte_sink() { }

        virtual size_t write(const byte* buffer, size_t n) = 0;
};

class Image {
    public:
        virtual ~Image() { }

        virtual void set_size(int w, int h, int d=-1) = 0;
        virtual void* rowp(int r) = 0;

        template<typename T>
        T* rowp_as(const int r) {
            return static_cast<T*>(this->rowp(r));
        }
};

class ImageFormat {
    public:
        virtual ~ImageFormat() { }

        virtual bool can_read() const { return false; }
        virtual bool can_write() const { return false; }

        virtual void read(byte_source* src, Image* output) {
            throw NotImplementedError();
        }
        virtual void write(Image* input, byte_sink* output) {
            throw NotImplementedError();
        }
};

#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
