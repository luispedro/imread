# -*- coding: utf-8 -*-
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
# License: MIT

from __future__ import division
try:
    import setuptools
except:
    print '''
setuptools not found.

On linux, the package is often called python-setuptools'''
    from sys import exit
    exit(1)
import os
import numpy.distutils.core as numpyutils


execfile('imread/imread_version.py')
long_description = file('docs/source/readme.rst').read()

undef_macros=[]
if os.environ.get('DEBUG'):
    undef_macros=['NDEBUG']

extensions = {
    'imread._imread': [
        'imread/_imread.cpp',
        'imread/lib/formats.cpp',
        'imread/lib/_png.cpp',
        'imread/lib/_jpeg.cpp',
        'imread/lib/_tiff.cpp',
        ],
}

ext_modules = [
    numpyutils.Extension(
        key,
        libraries=['png', 'jpeg', 'tiff'],
        sources=sources,
        undef_macros=undef_macros
        ) for key,sources in extensions.iteritems()]

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

