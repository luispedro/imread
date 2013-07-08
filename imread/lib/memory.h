// Copyright 2013 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#ifndef LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
#define LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013

#include "base.h"
#include <cstring>

class memory_source : public byte_source {
    public:
        memory_source(const byte* c, const int len)
            :data_(c)
            ,len_(len)
            ,pos_(0)
            { }
        ~memory_source() { }

        virtual size_t read(byte* buffer, size_t n) {
            if (pos_ + n > len_) n = len_-pos_;
            std::memmove(buffer, data_ + pos_, n);
            pos_ += n;
            return n;
        }
        virtual bool can_seek() const { return true; }
        virtual size_t seek_absolute(size_t pos) { return pos_ = pos; }
        virtual size_t seek_relative(int delta) { return pos_ += delta; }
        virtual size_t seek_end(int delta) { return pos_ = (len_-delta-1); }


    private:
        const byte* data_;
        const size_t len_;
        size_t pos_;
};

#endif // LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
