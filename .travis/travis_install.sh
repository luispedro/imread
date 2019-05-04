#!/usr/bin/env bash

set -e

if test -e $HOME/miniconda/envs/condaenv; then
    echo "condaenv already exists"
else
    conda create  --quiet --yes -n condaenv python=${TRAVIS_PYTHON_VERSION} numpy=${NUMPY_VERSION}
    conda install --quiet --yes -n condaenv jpeg libpng libwebp libtiff gcc_linux-64 gxx_linux-64 coveralls nose
fi

source activate condaenv
DEBUG=2 python setup.py build_ext --include-dirs=${CONDA_PREFIX}/include

