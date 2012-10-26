// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
#define LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012

#include "base.h"
#include <vector>

template<typename T>
inline std::vector<T*> allrows(Image& im) {
    std::vector<T*> res;
    const int h = im.dim(0);
    for (int r = 0; r != h; ++r) {
        res.push_back(im.rowp_as<T>(r));
    }
    return res;
}

inline std::vector<byte> full_data(byte_source& s) {
    std::vector<byte> res;
    byte buffer[4096];
    while (int n = s.read(buffer, sizeof buffer)) {
        res.insert(res.end(), buffer, buffer + n);
    }
    return res;
}

inline uint8_t read8(byte_source& s) {
    byte out;
    if (s.read(&out, 1) != 1) {
        throw CannotReadError("File ended prematurely");
    }
    return out;
}

inline uint16_t read16_le(byte_source& s) {
    uint8_t b0 = read8(s);
    uint8_t b1 = read8(s);
    return (uint16_t(b1) << 8)|uint16_t(b0);
}

inline uint32_t read32_le(byte_source& s) {
    uint16_t s0 = read16_le(s);
    uint16_t s1 = read16_le(s);
    return (uint32_t(s1) << 16)|uint32_t(s0);
}

#endif // LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
