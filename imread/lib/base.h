// Copyright 2012-2013 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <inttypes.h>
#include <memory>
#include <vector>
#include <assert.h>

#if defined(_MSC_VER)
 #include <io.h>
 #include <fcntl.h>
#else
 #include <unistd.h>
#endif

#include "errors.h"

#ifdef __GNUC__
#define warn_if_return_not_used __attribute__ ((__warn_unused_result__))
#endif

#ifndef warn_if_return_not_used
# define warn_if_return_not_used
#endif


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
        virtual size_t read(byte* buffer, size_t) warn_if_return_not_used = 0;
        void read_check(byte* buffer, size_t n) {
            if (this->read(buffer, n) != n) {
                throw CannotReadError("File ended prematurely");
            }
        }
#ifdef _GLIBCXX_DEBUG
        template <size_t Nelems>
        size_t read(byte (&arr)[Nelems], size_t n) {
            assert(n <= Nelems);
            byte* p = arr;
            return this->read(p, n);
        }
#endif
};

class byte_sink : virtual public seekable {
    public:
        virtual ~byte_sink() { }

        virtual size_t write(const byte* buffer, size_t n) warn_if_return_not_used = 0;
#ifdef _GLIBCXX_DEBUG
        template <size_t Nelems>
        size_t write(byte (&arr)[Nelems], size_t n) {
            assert(n <= Nelems);
            byte* p = arr;
            return this->write(p, n);
        }
#endif
        void write_check(const byte* buffer, size_t n) {
            if (this->write(buffer, n) != n) {
                throw CannotWriteError("Writing failed");
            }
        }
        virtual void flush() { }
};

class Image {
    public:
        virtual ~Image() { }

        virtual void* rowp(int r) = 0;

        virtual int nbits() const = 0;

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
    public:
        virtual ~ImageFactory() { }
        virtual std::auto_ptr<Image>
            create(int nbits, int d0, int d1, int d2, int d3=-1, int d4=-1) = 0;
    protected:
};

class ImageWithMetadata {
    public:
        ImageWithMetadata():meta_(0) { }
        virtual ~ImageWithMetadata() { delete meta_; };
        std::string* get_meta() { return meta_; }
        void set_meta(const std::string& m) { if (meta_) delete meta_; meta_ = new std::string(m); }
    private:
        std::string* meta_;
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
        virtual bool can_write_metadata() const { return false; }

        virtual std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory) {
            throw NotImplementedError();
        }
        virtual std::auto_ptr<image_list> read_multi(byte_source* src, ImageFactory* factory) {
            throw NotImplementedError();
        }
        virtual void write(Image* input, byte_sink* output) {
            throw NotImplementedError();
        }
        virtual void write_with_metadata(Image* input, byte_sink* output, const char*) {
            this->write(input, output);
        }
};

#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
