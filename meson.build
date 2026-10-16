project(
    'imread',
    'cpp',
    version: run_command(
        find_program('python', 'python3'),
        files('build-aux/meson-helper.py'),
        'version',
        check: true,
    ).stdout().strip(),
    license: 'MIT',
    meson_version: '>=1.2.3',
)

py = import('python').find_installation(pure: false)
helper = files('build-aux/meson-helper.py')

numpy_include_dir = run_command(
    py,
    helper,
    'numpy-include',
    check: true,
).stdout().strip()

cpp = meson.get_compiler('cpp')

defines = [
    'NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    'PY_ARRAY_UNIQUE_SYMBOL=MahotasImread_PyArray_API_Symbol',
]
if get_option('glibcpp_debug')
    defines += ['_GLIBCXX_DEBUG=1']
endif

extension_sources = [
    'imread/_imread.cpp',
    'imread/lib/formats.cpp',
    'imread/lib/numpy.cpp',
    'imread/lib/_bmp.cpp',
    'imread/lib/_jpeg.cpp',
    'imread/lib/_lsm.cpp',
    'imread/lib/_png.cpp',
    'imread/lib/_tiff.cpp',
]

deps = [
    dependency('libpng'),
    dependency('libjpeg'),
    dependency('libtiff-4'),
]

webp_dep = dependency('libwebp', required: get_option('webp'))
if webp_dep.found()
    extension_sources += ['imread/lib/_webp.cpp']
    deps += [webp_dep]
else
    defines += ['IMREAD_EXCLUDE_WEBP=1']
endif

if cpp.get_argument_syntax() == 'msvc'
    cpp_args = ['/EHsc']
    foreach d : defines
        cpp_args += ['/D' + d]
    endforeach
else
    cpp_args = []
    foreach d : defines
        cpp_args += ['-D' + d]
    endforeach
endif

python_sources = files(
    'imread/__init__.py',
    'imread/ijrois.py',
    'imread/imread.py',
    'imread/imread_version.py',
    'imread/special.py',
    'imread/tests/__init__.py',
    'imread/tests/test_bad.py',
    'imread/tests/test_bmp.py',
    'imread/tests/test_error.py',
    'imread/tests/test_formats.py',
    'imread/tests/test_imread.py',
    'imread/tests/test_imread_from_blob.py',
    'imread/tests/test_imsave.py',
    'imread/tests/test_jpeg.py',
    'imread/tests/test_png.py',
    'imread/tests/test_rois.py',
    'imread/tests/test_tiff.py',
    'imread/tests/test_webp.py',
    'imread/tests/test_xcf.py',
)

py.install_sources(
    python_sources,
    pure: false,
    preserve_path: true,
)

install_subdir(
    'imread/tests/data',
    install_dir: py.get_install_dir() / 'imread' / 'tests',
)

py.extension_module(
    '_imread',
    extension_sources,
    cpp_args: cpp_args,
    include_directories: include_directories(numpy_include_dir),
    dependencies: deps,
    install: true,
    subdir: 'imread',
)
