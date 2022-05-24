#!/usr/bin/env bash

# Based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -ev

PLAT=manylinux2014_x86_64

yum -y install libjpeg-devel libpng-devel libtiff-devel libwebp-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install imread --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" imread)
done

