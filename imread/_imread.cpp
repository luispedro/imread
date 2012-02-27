// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#include "lib/base.h"
#include "lib/formats.h"
#include "lib/file.h"
#include "lib/numpy.h"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _imread (which is dangerous: types are not checked!) or a bug in imread.py.\n";

PyObject* py_imread(PyObject* self, PyObject* args) {
    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    int fd = ::open(filename, O_RDONLY);
    if (fd < 0) {
        PyErr_SetString(PyExc_OSError, "File does not exist");
        return 0;
    }
    const char* formatstr = strrchr(filename, '.');
    if (!formatstr) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read extension");
        return NULL;
    }
    ++formatstr;

    try {
        NumpyFactory factory;
        std::auto_ptr<byte_source> input(new fd_source_sink(fd));
        std::auto_ptr<ImageFormat> format(get_format(formatstr));
        if (!format.get() || !format->can_read()) {
            throw CannotReadError("Cannot read this format");
        }
        std::auto_ptr<Image> output = format->read(input.get(), &factory);
        return static_cast<NumpyImage&>(*output).releasePyObject();

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Mysterious error");
        return 0;
    }
}

PyObject* py_imsave(PyObject* self, PyObject* args) {
    const char* filename;
    const char* formatstr;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "ssO", &filename, &formatstr, &array) || !PyArray_Check(array)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    int fd = ::open(filename, O_CREAT|O_WRONLY, 0644);
    if (fd < 0) {
        PyErr_SetString(PyExc_OSError, "File does not exist");
        return 0;
    }

    Py_INCREF(array);
    try {
        NumpyImage input(array);
        std::auto_ptr<byte_sink> output(new fd_source_sink(fd));
        std::auto_ptr<ImageFormat> format(get_format(formatstr));
        if (!format->can_write()) {
            throw CannotWriteError("Cannot write this format");
        }
        format->write(&input, output.get());
        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Mysterious error");
        return 0;
    }
}


PyMethodDef methods[] = {
  {"imread",(PyCFunction)py_imread, METH_VARARGS, NULL},
  {"imsave",(PyCFunction)py_imsave, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_imread()
  {
    import_array();
    (void)Py_InitModule("_imread", methods);
  }

