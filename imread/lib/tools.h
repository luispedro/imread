// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
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

struct stack_based_memory_pool {
    // This class manages an allocator which releases all allocated memory on
    // stack exit
    stack_based_memory_pool() { }
    ~stack_based_memory_pool() {
        for (unsigned i = 0; i != data_.size(); ++i) {
            operator delete(data_[i]);
            data_[i] = 0;
        }
    }

    void* allocate(const int n) {
        data_.reserve(data_.size() + 1);
        void* d = operator new(n);
        data_.push_back(d);
        return d;
    }

    template <typename T>
    T allocate_as(const int n) {
        return static_cast<T>(this->allocate(n));
    }

    private:
    std::vector<void*> data_;
};

#endif // LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
