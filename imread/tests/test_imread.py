from nose.tools import raises
import numpy as np

from imread import imread

def test_with_dot():
    f = imread('./imread/tests/data/good.png')
    assert f.shape == (2,2)

def test_uppercase():
    f = imread('./imread/tests/data/GOOD.PNG')
    assert f.shape == (2,2)

@raises(ValueError)
def test_no_ext():
    imread('file_without_extension')


def test_formatstr():
    f = imread('./imread/tests/data/good', formatstr='png')
    assert f.shape == (2,2)

