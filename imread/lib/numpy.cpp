#include <cstring>
#include <vector>
#include "numpy.h"
void NumpyImage::finalize() {
    if (PyArray_TYPE(array_) == NPY_BOOL) {
        // We need to expand this
        const int h = PyArray_DIM(array_, 0);
        const int w = PyArray_DIM(array_, 1);
        std::vector<byte> buf;
        buf.resize(w);
        for (int y = 0; y != h; ++y) {
            uint8_t* data = static_cast<uint8_t*>(PyArray_GETPTR1(array_, y));
            for (int x = 0; x != ((w/8)+bool(w%8)); ++x) {
                const uint8_t v = data[x];
                for (int b = 0; b != 8 && (x*8+b < w); ++b) {
                    buf[x*8+b] = bool(v & (1 << (7-b)));
                }
            }
            std::memcpy(data, &buf[0], w);
        }
    }
}

