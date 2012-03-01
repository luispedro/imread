// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <inttypes.h>
#include <memory>
#include <vector>

#if defined(_MSC_VER)
 #include <io.h>
 #include <fcntl.h>
#else
 #include <unistd.h>
#endif

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
        virtual std::auto_ptr<Image>
            create(int nbits, int w, int h, int d) = 0;

    protected:
};


/// This class *owns* its members and will delete them if destroyed
struct image_list {
    public:
        image_list() { }
        ~image_list() {
            for (unsigned i = 0; i != content.size(); ++i) delete content[i];
        }
        std::vector<Image*>::size_type size() const { return content.size(); }
        void push_back(std::auto_ptr<Image> p) { content.push_back(p.release()); }

        /// After release(), all of the pointers will be owned by the caller
        /// who must figure out how to delete them. Note that release() resets the list.
        std::vector<Image*> release() {
            std::vector<Image*> r;
            r.swap(content);
            return r;
        }
    private:
        image_list(const image_list&);
        image_list& operator = (const image_list&);
        std::vector<Image*> content;
};


class ImageFormat {
    public:
        virtual ~ImageFormat() { }

        virtual bool can_read() const { return false; }
        virtual bool can_read_multi() const { return false; }
        virtual bool can_write() const { return false; }

        virtual std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory) {
            throw NotImplementedError();
        }
        virtual std::auto_ptr<image_list> read_multi(byte_source* src, ImageFactory* factory) {
            throw NotImplementedError();
        }
        virtual void write(Image* input, byte_sink* output) {
            throw NotImplementedError();
        }
};

#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
