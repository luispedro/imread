from nose.tools import with_setup, raises
import numpy as np
from imread import imread, imsave
from imread import _imread
import numpy as np

_filename = 'imread_testing_file.png'

def _remove_file():
    from os import unlink
    try:
        unlink(_filename)
    except:
        pass

@with_setup(teardown=_remove_file)
def test_png_raw():
    simple = np.arange(16*16).reshape((16,16))
    simple = simple.astype(np.uint8)
    _imread.imsave(_filename, 'png', simple)
    back = imread(_filename)
    assert np.all(simple == back)

@with_setup(teardown=_remove_file)
def test_asym():
    simple = np.arange(16*16).reshape((32,8))
    simple = simple.astype(np.uint8)
    _imread.imsave(_filename, 'png', simple)
    back = imread(_filename)
    assert np.all(simple == back)

@with_setup(teardown=_remove_file)
def test_random():
    np.random.seed(23)
    for i in range(8):
        simple = np.random.random_sample((64,64,3))
        simple *= 255
        simple = simple.astype(np.uint8)
        if i < 3:
            simple[:,:,i] = 0
        _imread.imsave(_filename, 'png', simple)
        back = imread(_filename)
        assert np.all(simple == back)


@with_setup(teardown=_remove_file)
def test_non_carray():
    np.random.seed(87)
    simple = np.random.random_sample((128,128,3))
    simple *= 255
    simple = simple.astype(np.uint8)
    simple = simple[32:64,32::2]
    imsave(_filename, simple, 'png')
    back = imread(_filename)
    assert np.all(simple == back)


def test_binary():
    f = imread('imread/tests/data/bit1.png')
    assert f.dtype == np.bool_

@raises(RuntimeError)
def test_error():
    imread('imread/tests/data/error.png')

def test_regression():
    im = imread('imread/tests/data/palette_zero.png')
    assert im.sum() == 0
    assert im.shape == (128, 64, 3)
