// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)


#if defined(_MSC_VER)
 #include <io.h>
#else
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/stat.h>
 #include <fcntl.h>
#endif


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

PyObject* py_imread_may_multi(PyObject* self, PyObject* args, bool is_multi) {
    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    const char* formatstr = strrchr(filename, '.');
    if (!formatstr) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read extension");
        return NULL;
    }
    ++formatstr;

    int fd = ::open(filename, O_RDONLY);
    if (fd < 0) {
        PyErr_SetString(PyExc_OSError, "File does not exist");
        return 0;
    }

    try {
        std::auto_ptr<ImageFormat> format(get_format(formatstr));
        if (!format.get()) {
            throw CannotReadError("This format is unknown to imread");
        }
        if (is_multi && !format->can_read_multi()) {
            throw CannotReadError("imread cannot read_multi in this format");
        }
        if (!is_multi && !format->can_read()) {
            if (format->can_read_multi()) {
                throw CannotReadError("imread cannot read in this format (but can read_multi!)");
            } else {
                throw CannotReadError("imread cannot read in this format");
            }
        }

        NumpyFactory factory;
        std::auto_ptr<byte_source> input(new fd_source_sink(fd));
        if (is_multi) {
            std::auto_ptr<image_list> images = format->read_multi(input.get(), &factory);
            PyObject* output = PyList_New(images->size());
            if (!output) return NULL;
            std::vector<Image*> pages = images->release();
            for (unsigned i = 0; i != pages.size(); ++i) {
                PyList_SET_ITEM(output, i, static_cast<NumpyImage&>(*pages[i]).releasePyObject());
                delete pages[i];
            }
            return output;
        } else {
            std::auto_ptr<Image> output = format->read(input.get(), &factory);
            return static_cast<NumpyImage&>(*output).releasePyObject();
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Mysterious error");
        return 0;
    }
}

PyObject* py_imread      (PyObject* self, PyObject* args) { return py_imread_may_multi(self, args, false); }
PyObject* py_imread_multi(PyObject* self, PyObject* args) { return py_imread_may_multi(self, args, true); }

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
  {"imread_multi",(PyCFunction)py_imread_multi, METH_VARARGS, NULL},
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

