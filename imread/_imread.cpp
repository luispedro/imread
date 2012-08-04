// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)


#if defined(_MSC_VER)
 #include <io.h>
#else
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/stat.h>
 #include <fcntl.h>
 const int O_BINARY = 0;
#endif

#include <sstream>

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
    const char* formatstr;
    if (!PyArg_ParseTuple(args, "ss", &filename, &formatstr)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }

    int fd = ::open(filename, O_RDONLY|O_BINARY);
    if (fd < 0) {
        std::stringstream ss;
        ss << "File `" << filename << "` does not exist";
        PyErr_SetString(PyExc_OSError, ss.str().c_str());
        return 0;
    }

    try {
        std::auto_ptr<ImageFormat> format(get_format(formatstr));
        if (!format.get()) {
            std::stringstream ss;
            ss << "This format (" << formatstr << ") is unknown to imread";
            throw CannotReadError(ss.str());
        }
        if (is_multi && !format->can_read_multi()) {
            std::stringstream ss;
            ss << "imread cannot read_multi in this format (" << formatstr << ")";
            if (format->can_read()) {
                ss << " but read() will work.";
            }
            throw CannotReadError(ss.str());
        }
        if (!is_multi && !format->can_read()) {
            std::stringstream ss;
            ss << "imread cannot read_in this format (" << formatstr << ")";
            if (format->can_read_multi()) {
                ss << "(but can read_multi!)";
            }
            throw CannotReadError(ss.str());
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
    int fd = ::open(filename, O_CREAT|O_WRONLY|O_BINARY|O_TRUNC, 0644);
    if (fd < 0) {
        std::stringstream ss;
        ss << "Cannot open file '" << filename << "' for writing";
        PyErr_SetString(PyExc_OSError, ss.str().c_str());
        return 0;
    }

    Py_INCREF(array);
    try {
        NumpyImage input(array);
        std::auto_ptr<byte_sink> output(new fd_source_sink(fd));
        std::auto_ptr<ImageFormat> format(get_format(formatstr));
        if (!format->can_write()) {
            std::stringstream ss;
            ss << "Cannot write this format (" << formatstr << ")";
            throw CannotWriteError(ss.str());
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

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC
init_imread()
  {
    import_array();
    (void)Py_InitModule("_imread", methods);
  }
#else

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_imread",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC
PyInit__imread()
  {
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
  }
#endif
