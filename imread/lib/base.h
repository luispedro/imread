// Copyright 2012-2015 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <inttypes.h>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <map>
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

        virtual std::unique_ptr<Image> clone() const = 0;

        virtual void* rowp(int r) = 0;

        virtual int nbits() const = 0;
        int nbytes() const {
            const int bits = this->nbits();
            return (bits / 8) + bool(bits % 8);
        }

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
        virtual std::unique_ptr<Image>
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
        void push_back(std::unique_ptr<Image>&& p) { content.push_back(p.release()); }
        Image* at(const unsigned ix) const { return content.at(ix); }

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


/// number_or_string is a sort of typed union.
/// We could have used boost::any here, but that would have brought in a big
/// dependency, which would otherwise not be used.
struct number_or_string {
    number_or_string()
        :holds_(ns_empty)
        { }

    explicit number_or_string(std::string s)
        :str_(s)
        ,holds_(ns_string)
        { }
    explicit number_or_string(int i)
        :int_(i)
        ,holds_(ns_int)
        { }
    explicit number_or_string(double v)
        :double_(v)
        ,holds_(ns_double)
        { }

    bool get_int(int& n) const { if (holds_ != ns_int) return false; n = int_; return true; }
    bool get_double(double& n) const { if (holds_ != ns_double) return false; n = double_; return true; }
    bool get_str(std::string& s) const { if (holds_ != ns_string) return false; s = str_; return true; }
    const char* maybe_c_str() const {
        if (holds_ == ns_string) return str_.c_str();
        return 0;
    }

    private:
        std::string str_;
        int int_;
        double double_;
        enum { ns_empty, ns_string, ns_int, ns_double } holds_;
};


typedef std::map<std::string, number_or_string> options_map;

inline
const char* get_optional_cstring(const options_map& opts, const std::string key) {
    options_map::const_iterator iter = opts.find(key);
    if (iter == opts.end()) return 0;
    return iter->second.maybe_c_str();
}

inline
int get_optional_int(const options_map& opts, const std::string key, const int def) {
    options_map::const_iterator iter = opts.find(key);
    if (iter == opts.end()) return def;
    int v;
    if (iter->second.get_int(v)) { return v; }
    return def;
}

inline
bool get_optional_bool(const options_map& opts, const std::string key, const bool def) {
    return get_optional_int(opts, key, def);
}

inline
bool match_magic(byte_source* src, const char* magic, const size_t n) {
    if (!src->can_seek()) return false;
    std::vector<byte> buf;
    buf.resize(n);
    const size_t n_read = src->read(&buf.front(), n);
    src->seek_relative(-n_read);

    return (n_read == n && std::memcmp(&buf.front(), magic, n) == 0);
}

class ImageFormat {
    public:
        virtual ~ImageFormat() { }

        virtual bool can_read() const { return false; }
        virtual bool can_read_multi() const { return false; }
        virtual bool can_write() const { return false; }
        virtual bool can_write_multi() const { return false; }
        virtual bool can_write_metadata() const { return false; }

        virtual std::unique_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map&) {
            throw NotImplementedError();
        }
        virtual std::unique_ptr<image_list> read_multi(byte_source* src, ImageFactory* factory, const options_map&) {
            throw NotImplementedError();
        }
        virtual void write(Image* input, byte_sink* output, const options_map&) {
            throw NotImplementedError();
        }
        virtual void write_multi(image_list* input, byte_sink* output, const options_map&) {
            throw NotImplementedError();
        }
};



#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
