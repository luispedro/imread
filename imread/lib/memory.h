// Copyright 2013 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#ifndef LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
#define LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013

#include "base.h"
#include <cstring>

// Seeks follow lseek() semantics: seeking past the end is allowed (later reads
// return 0 bytes), but seeking to before the start throws CannotSeekError.
class memory_source : public byte_source {
    public:
        memory_source(const byte* c, const size_t len)
            :data_(c)
            ,len_(len)
            ,pos_(0)
            { }
        ~memory_source() { }

        virtual size_t read(byte* buffer, size_t n) {
            if (pos_ >= len_) return 0;
            if (n > len_ - pos_) n = len_ - pos_;
            std::memmove(buffer, data_ + pos_, n);
            pos_ += n;
            return n;
        }
        virtual bool can_seek() const { return true; }
        virtual size_t seek_absolute(size_t pos) { return pos_ = pos; }
        virtual size_t seek_relative(int delta) { return pos_ = offset_from(pos_, delta); }
        virtual size_t seek_end(int delta) { return pos_ = offset_from(len_, delta); }


    private:
        static size_t offset_from(size_t base, int delta) {
            if (delta < 0 && size_t(-(long long)delta) > base) {
                throw CannotSeekError("imread: attempted to seek before the start of the blob");
            }
            return base + delta;
        }
        const byte* data_;
        const size_t len_;
        size_t pos_;
};

#endif // LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
