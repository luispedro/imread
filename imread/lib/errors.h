// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <exception>

struct NotImplementedError : std::exception {
    NotImplementedError()
        :w(0)
        { }
    NotImplementedError(const char* w)
        :w(w)
        { }

    const char* what() const throw() { return w ? w : "Not Implemented"; }

    const char* w;
};

struct ProgrammingError : std::exception {
    ProgrammingError()
        :w(0)
        { }
    ProgrammingError(const char* w)
        :w(w)
        { }

    const char* what() const throw() { return w ? w : "Programming Error"; }

    const char* w;
};

struct CannotReadError : std::exception {
    CannotReadError(const char* w)
        :w(w)
        { }
    CannotReadError()
        :w(0)
        { }

    const char* what() const throw() { return w ? w : "Read Error"; }
    const char* w;

};
struct CannotWriteError : std::exception {
    CannotWriteError(const char* w)
        :w(w)
        { }
    CannotWriteError()
        :w(0)
        { }

    const char* what() const throw() { return w ? w : "Write Error"; }
    const char* w;
};

#endif // LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
