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

#endif // LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
