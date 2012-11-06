# -*- coding: utf-8 -*-
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
# License: MIT

from __future__ import division, print_function
import sys

try:
    import setuptools
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    sys.exit(1)

import os
import numpy.distutils.core as numpyutils


exec(compile(open('imread/imread_version.py').read(), 
             'imread/imread_version.py', 'exec'))
long_description = open('README.rst').read()

undef_macros = []
define_macros = []
if os.environ.get('DEBUG'):
    undef_macros = ['NDEBUG']
    if os.environ.get('DEBUG') == '2':
        define_macros = [('_GLIBCXX_DEBUG','1')]

include_dirs = []
library_dirs = []

for pth in ('/usr/local/include', '/usr/X11/include'):
    if os.path.isdir(pth):
        include_dirs.append(pth)

for pth in ('/usr/local/lib', '/usr/X11/lib'):
    if os.path.isdir(pth):
        library_dirs.append(pth)

extensions = {
    'imread._imread': [
        'imread/_imread.cpp',
        'imread/lib/formats.cpp',
        'imread/lib/numpy.cpp',
        'imread/lib/_bmp.cpp',
        'imread/lib/_jpeg.cpp',
        'imread/lib/_lsm.cpp',
        'imread/lib/_png.cpp',
        'imread/lib/_tiff.cpp',
        'imread/lib/_webp.cpp',
        ],
}

libraries = ['png', 'jpeg', 'tiff', 'webp']
if sys.platform.startswith('win'):
    libraries.append('zlib')

ext_modules = [
    numpyutils.Extension(
        key,
        libraries = libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        sources=sources,
        undef_macros=undef_macros,
        define_macros=define_macros,
        ) for key, sources in extensions.items()]

packages = setuptools.find_packages()

package_dir = {
    'imread.tests': 'imread/tests',
    }
package_data = {
    'imread.tests': ['data/*'],
    }

classifiers = [
'Development Status :: 4 - Beta',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Topic :: Multimedia',
'Topic :: Scientific/Engineering :: Image Recognition',
'Topic :: Software Development :: Libraries',
'Programming Language :: Python',
'Programming Language :: Python :: 3',
'Programming Language :: C++',
'License :: OSI Approved :: MIT License',
]

numpyutils.setup(name = 'imread',
      version = __version__,
      description = 'imread: Image reading library',
      long_description = long_description,
      author = 'Luis Pedro Coelho',
      author_email = 'luis@luispedro.org',
      license = 'MIT',
      platforms = ['Any'],
      classifiers = classifiers,
      url = 'http://luispedro.org/software/imread',
      packages = packages,
      ext_modules = ext_modules,
      package_dir = package_dir,
      package_data = package_data,
      test_suite = 'nose.collector',
      )
