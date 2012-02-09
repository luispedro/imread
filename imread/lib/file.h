// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <sys/types.h>
#include <unistd.h>

#include "base.h"

class fd_source_sink : public byte_source, public byte_sink {
    public:
        fd_source_sink(int fd)
            :fd_(fd)
            { }
        ~fd_source_sink() {
            ::close(fd_);
        }
        virtual size_t read(byte* buffer, size_t n) {
            return ::read(fd_, buffer, n);
        }
        virtual bool can_seek() const { return true; }
        virtual size_t seek_absolute(size_t pos) { return ::lseek(fd_, pos, SEEK_SET); }
        virtual size_t seek_relative(int delta) { return ::lseek(fd_, delta, SEEK_CUR); }
        virtual size_t seek_end(int delta) { return ::lseek(fd_, delta, SEEK_END); }


        virtual size_t write(const byte* buffer, size_t n) {
            return ::write(fd_, buffer, n);
        }
    private:
        int fd_;
};

#endif // LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
