=============================
How To Install Mahotas-imread
=============================

From source
-----------

You can get the released version using our favorite Python package manager::

    pip install imread

If you prefer, you can download the source from `PyPI
<http://pypi.python.org/pypi/mahotas>`__ and run::

    python setup.py install

You will need to have ``numpy`` and a ``C++`` compiler.


Bleeding Edge (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Development happens on `github <https://github.com/luispedro/imread>`__. You
can get the development source there. Watch out that *these versions are more
likely to have problems*.

On Windows
----------

On Windows, Christoph Gohlke does an excelent job maintaining `binary packages
of imread <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__ (and several other
packages).

conda
~~~~~

Imread is not a part of standard conda packages, but on 64 bit Linux, you can
get it `from this repository <https://binstar.org/luispedro/imread`__ with::

    conda install -c https://conda.binstar.org/luispedro imread

