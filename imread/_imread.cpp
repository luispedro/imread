// Copyright 2012-2019 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)


#if defined(_MSC_VER)
 #include <io.h>
#else
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/stat.h>
 #include <fcntl.h>
 #ifndef O_BINARY
  const int O_BINARY = 0;
 #endif
#endif

#include <sstream>
#include <iostream>

#include "lib/base.h"
#include "lib/formats.h"
#include "lib/file.h"
#include "lib/memory.h"
#include "lib/numpy.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <errno.h>

namespace{

struct py_exception_set { };

const char* get_blob(PyObject* data, size_t& len) {
#if PY_MAJOR_VERSION < 3
    if (!PyString_Check(data)) return 0;
    len = PyString_Size(data);
    return PyString_AsString(data);
#else
    len = PyBytes_Size(data);
    if (!PyBytes_Check(data)) return 0;
    return PyBytes_AsString(data);
#endif
}


const char* get_cstring(PyObject* stro) {
#if PY_MAJOR_VERSION < 3
    if (!PyString_Check(stro)) return 0;
    return PyString_AsString(stro);
#else
    if (!PyUnicode_Check(stro)) return 0;
    return PyUnicode_AsUTF8(stro);
#endif
}


options_map parse_options(PyObject* dict) {
    options_map res;
    if (!PyDict_Check(dict)) return res;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(dict, &pos, &key, &value)) {
        std::string k = get_cstring(key);
        if (PyLong_Check(value)) {
            res[k] = number_or_string(int(PyLong_AsLong(value)));
#if PY_MAJOR_VERSION < 3
        } else if (PyInt_Check(value)) {
            res[k] = number_or_string(int(PyInt_AS_LONG(value)));
#endif
        } else if (PyFloat_Check(value)) {
            res[k] = number_or_string(PyFloat_AS_DOUBLE(value));
#if PY_MAJOR_VERSION >= 3
        } else if (PyBytes_Check(value)) {
            size_t len;
            const char* blob = get_blob(value, len);
            res[k] = number_or_string(std::string(blob, len));
#endif
        } else {
            const char* c = get_cstring(value);
            if (!c) {
                std::stringstream ss;
                ss << "Error while parsing option value for '" << k << "': type was not understood.";
                throw OptionsError(ss.str());
            }
            res[k] = number_or_string(std::string(c));
        }
    }
    return res;
}

std::unique_ptr<byte_source> get_input(PyObject* filename_or_blob_object, const bool is_blob) {
    if (is_blob) {
        size_t len;
        const char* data = get_blob(filename_or_blob_object, len);
        if (!data) {
            PyErr_SetString(PyExc_TypeError, "Expected blob of bytes");
            throw py_exception_set();
        }
        return std::unique_ptr<byte_source>(new memory_source(reinterpret_cast<const byte*>(data), len));
    } else {
        const char* filename = get_cstring(filename_or_blob_object);
        if (!filename) throw py_exception_set();
        int fd = ::open(filename, O_RDONLY|O_BINARY);
        if (fd < 0) {
            std::stringstream ss;
            if (errno == EACCES) {
                ss << "Permission error when opening `" << filename << "`";
            } else if (errno == ENOENT) {
                ss << "File `" << filename << "` does not exist";
            } else {
                ss << "Unknown error opening `" << filename << "`.";
            }
            PyErr_SetString(PyExc_OSError, ss.str().c_str());
            throw py_exception_set();
        }
        return std::unique_ptr<byte_source>(new fd_source_sink(fd));
    }
}

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _imread (which is dangerous: types are not checked!) or a bug in imread.py.\n";

PyObject* py_detect_format(PyObject* self, PyObject* args) {
    PyObject* filename_or_blob_object;
    int is_blob;
    if (!PyArg_ParseTuple(args, "Oi", &filename_or_blob_object, &is_blob)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }

    try {
        std::unique_ptr<byte_source> input = get_input(filename_or_blob_object, is_blob);
        const char* format = magic_format(&*input);
        if (!format) Py_RETURN_NONE;
#if PY_MAJOR_VERSION >= 3
        return PyUnicode_FromString(format);
#else
        return PyBytes_FromString(format);
#endif
    } catch (py_exception_set) {
        return 0;
    } catch (const std::bad_alloc& a) {
        PyErr_SetString(PyExc_MemoryError, a.what());
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Mysterious error");
        return 0;
    }
}


PyObject* py_supports_format(PyObject* self, PyObject* args) {
    const char* formatstr;
    if (!PyArg_ParseTuple(args, "s", &formatstr)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    const bool r = get_format(formatstr).get();
    if (r) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}


PyObject* py_imread_may_multi(PyObject* self, PyObject* args, bool is_multi, bool is_blob) {
    PyObject* filename_or_blob_object;
    const char* formatstr;
    PyObject* optsDict;
    if (!PyArg_ParseTuple(args, "OsO", &filename_or_blob_object, &formatstr, &optsDict)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    options_map opts = parse_options(optsDict);

    try {
        std::unique_ptr<ImageFormat> format(get_format(formatstr));
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
        std::unique_ptr<byte_source> input = get_input(filename_or_blob_object, is_blob);
        if (is_multi) {
            std::unique_ptr<image_list> images = format->read_multi(input.get(), &factory, opts);
            PyObject* output = PyList_New(images->size());
            if (!output) return NULL;
            std::vector<Image*> pages = images->release();
            for (unsigned i = 0; i != pages.size(); ++i) {
                PyList_SET_ITEM(output, i, static_cast<NumpyImage&>(*pages[i]).releasePyObject());
                delete pages[i];
            }
            return output;
        } else {
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            PyObject* final = PyTuple_New(2);
            if (!final) return NULL;
            PyTuple_SET_ITEM(final, 0, static_cast<NumpyImage&>(*output).releasePyObject());
            PyTuple_SET_ITEM(final, 1, static_cast<NumpyImage&>(*output).metadataPyObject());
            return final;
        }
    } catch (py_exception_set) {
        return 0;
    } catch (const std::bad_alloc& a) {
        PyErr_SetString(PyExc_MemoryError, a.what());
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "imread: Unknown error occurred (please report a bug at https://github.com/luispedro/imread/issues).");
        return 0;
    }
}

PyObject* py_imread             (PyObject* self, PyObject* args) { return py_imread_may_multi(self, args, false, false); }
PyObject* py_imread_multi       (PyObject* self, PyObject* args) { return py_imread_may_multi(self, args,  true, false); }
PyObject* py_imread_from_blob   (PyObject* self, PyObject* args) { return py_imread_may_multi(self, args, false, true); }

PyObject* py_imsave_may_multi(PyObject* self, PyObject* args, bool is_multi) {
    const char* filename;
    const char* formatstr;
    PyObject* formatstrObject;
    PyObject* arrays;
    PyArrayObject* array = 0;
    PyObject* optsDict;
    if (!PyArg_ParseTuple(args, "sOOO", &filename, &formatstrObject, &arrays, &optsDict)) return NULL;
    if (is_multi) {
        if (!PyList_Check(arrays)) {
            PyErr_SetString(PyExc_RuntimeError, "List expected for imsave_multi");
            return NULL;
        }
    } else {
        array = (PyArrayObject*)arrays;
        if (!PyArray_Check(array)) {
            PyErr_SetString(PyExc_RuntimeError, "array expected for imsave");
            return NULL;
        }
    }

    formatstr = get_cstring(formatstrObject);
    if (!formatstr) {
        PyErr_SetString(PyExc_TypeError, "imread.imsave: Expected a string as formatstr");
        return NULL;
    }
    try {
        options_map opts = parse_options(optsDict);
        std::unique_ptr<ImageFormat> format(get_format(formatstr));
        if (!format.get()) {
            std::stringstream ss;
            ss << "Handler not found for format '" << formatstr << "'";
            throw CannotWriteError(ss.str());
        }
        if (!is_multi && !format->can_write()) {
            std::stringstream ss;
            ss << "Cannot write this format (format: " << formatstr << ")";
            throw CannotWriteError(ss.str());
        }
        if (is_multi && !format->can_write_multi()) {
            std::stringstream ss;
            ss << "Cannot write multiple pages with this format (format: " << formatstr << ")";
            throw CannotWriteError(ss.str());
        }

        const int fd = ::open(filename, O_CREAT|O_RDWR|O_BINARY|O_TRUNC, 0644);
        if (fd < 0) {
            std::stringstream ss;
            ss << "Cannot open file '" << filename << "' for writing";
            throw CannotWriteError(ss.str());
        }

        std::unique_ptr<byte_sink> output(new fd_source_sink(fd));

        if (is_multi) {
            image_list array_list;
            const unsigned n = PyList_GET_SIZE(arrays);
            for (int i = 0; i != int(n); ++i) {
                array = (PyArrayObject*)PyList_GET_ITEM(arrays, i);
                if (!PyArray_Check(array)) {
                    PyErr_SetString(PyExc_RuntimeError, "imsave_multi: Array expected in list");
                    return NULL;
                }
                // ~NumpyImage() will decrease the count
                Py_INCREF(array);
                NumpyImage* input = new NumpyImage(array);
                array_list.push_back(std::unique_ptr<Image>(input));
            }
            format->write_multi(&array_list, output.get(), opts);
        } else {
            // ~NumpyImage() will decrease the count
            Py_INCREF(array);
            NumpyImage input(array);
            format->write(&input, output.get(), opts);
        }
        Py_RETURN_NONE;
    } catch (py_exception_set) {
        return 0;
    } catch (const std::bad_alloc& a) {
        PyErr_SetString(PyExc_MemoryError, a.what());
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "imread: Unknown error occurred (please report a bug at https://github.com/luispedro/imread/issues).");
        return 0;
    }
}

PyObject* py_imsave       (PyObject* self, PyObject* args) { return py_imsave_may_multi(self, args, false); }
PyObject* py_imsave_multi (PyObject* self, PyObject* args) { return py_imsave_may_multi(self, args, true); }


PyMethodDef methods[] = {
  {"detect_format",(PyCFunction)py_detect_format, METH_VARARGS, NULL},
  {"supports_format",(PyCFunction)py_supports_format, METH_VARARGS, NULL},
  {"imread",(PyCFunction)py_imread, METH_VARARGS, NULL},
  {"imread_multi",(PyCFunction)py_imread_multi, METH_VARARGS, NULL},
  {"imread_from_blob",(PyCFunction)py_imread_from_blob, METH_VARARGS, NULL},
  {"imsave",(PyCFunction)py_imsave, METH_VARARGS, NULL},
  {"imsave_multi",(PyCFunction)py_imsave_multi, METH_VARARGS, NULL},
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
