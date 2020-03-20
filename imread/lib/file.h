// Copyright 2012-2020 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#ifndef LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#if defined(_MSC_VER)
 #include <io.h>
#else
 #include <unistd.h>
 #include <sys/types.h>
#endif

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
        virtual size_t seek_absolute(size_t pos) { return _check_seek(fd_, pos, SEEK_SET); }
        virtual size_t seek_relative(int delta) { return _check_seek(fd_, delta, SEEK_CUR); }
        virtual size_t seek_end(int delta) { return _check_seek(fd_, delta, SEEK_END); }


        virtual size_t write(const byte* buffer, size_t n) {
            return ::write(fd_, buffer, n);
        }
    private:
        int fd_;
        static off_t _check_seek(int fd, off_t offset, int whence) {
            off_t r = ::lseek(fd, offset, whence);
            if (r == static_cast<off_t>(-1)) {
                throw CannotSeekError();
            }
            return r;
        }
};

#endif // LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
