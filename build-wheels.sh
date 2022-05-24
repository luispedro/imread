#!/usr/bin/env bash

# Based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -ev

PLAT=manylinux2014_x86_64

yum -y install libjpeg-devel libpng-devel libtiff-devel libwebp-devel

# Compile wheels
for PYVER in \
       cp36-cp36m \
       cp37-cp37m \
       cp38-cp38 \
       cp39-cp39 \
       cp310-cp310 \
       ; do

    PYBIN="/opt/python/${PYVER}/bin"
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/imread*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

