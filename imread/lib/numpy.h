// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#include <memory>
#include "base.h"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

class NumpyImage : public Image {
    public:
        NumpyImage(PyArrayObject* array = 0)
            :array_(array)
            { }

        ~NumpyImage() {
            Py_XDECREF(array_);
        }

        PyArrayObject* release() {
            PyArrayObject* r = array_;
            array_ = 0;
            return r;
        }

        PyObject* releasePyObject() {
            return reinterpret_cast<PyObject*>(this->release());
        }

        void set_size(int w, int h, int d = -1) {
            if (array_ &&
                    PyArray_DIM(array_, 0) == h &&
                    PyArray_DIM(array_, 1) == w &&
                    ((d == -1 && PyArray_NDIM(array_) == 2) ||
                     (PyArray_NDIM(array_) == 3 && PyArray_DIM(array_, 2) == d))
                    ) return;

            Py_XDECREF(array_);
            npy_intp dims[3];
            dims[0] = h;
            dims[1] = w;
            dims[2] = d;
            const npy_intp nd = 2 + (d != -1);
            array_ = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims, NPY_UINT8));
            if (!array_) throw std::bad_alloc();
        }

        virtual int ndims() const {
            if (!array_) throw ProgrammingError();
            return PyArray_NDIM(array_);
        }
        virtual int dim(int d) const {
            if (!array_ || d >= this->ndims()) throw ProgrammingError();
            return PyArray_DIM(array_, d);
        }

        virtual void* rowp(int r) {
            if (!array_) throw ProgrammingError();
            if (r >= PyArray_DIM(array_, 0)) throw ProgrammingError();
            return PyArray_GETPTR1(array_, r);
        }
        PyArrayObject* array_;
};

#endif // LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
