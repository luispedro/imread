from nose.tools import with_setup, raises
import numpy as np
from imread import imread

def test_read():
    im = imread('imread/tests/data/star1.bmp')
    assert np.any(im)
    assert im.shape == (128, 128, 3)
