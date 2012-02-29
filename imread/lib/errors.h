// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <string>
#include <exception>

#define DECLARE_IMREAD_ERROR_TYPE(TypeName, DefaultMsg) \
    struct TypeName : std::exception { \
        TypeName(const char* e) \
            :w(e) \
            { } \
        TypeName(std::string e) \
            :w(e) \
            { } \
        TypeName() \
            :w(DefaultMsg) \
            { } \
        ~TypeName() throw() { } \
        \
        \
        const char* what() const throw() { return w.c_str(); } \
        \
        std::string w;\
    }

DECLARE_IMREAD_ERROR_TYPE(CannotReadError, "Read Error");
DECLARE_IMREAD_ERROR_TYPE(CannotWriteError, "Write Error");
DECLARE_IMREAD_ERROR_TYPE(NotImplementedError, "Not Implemented");
DECLARE_IMREAD_ERROR_TYPE(ProgrammingError, "Programming Error");

#endif // LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
