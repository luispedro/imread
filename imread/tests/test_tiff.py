from nose.tools import with_setup, raises
import numpy as np
from imread import imread, imsave
import numpy as np

_filename = 'imread_testing_file.tiff'

def _remove_file():
    from os import unlink
    try:
        unlink(_filename)
    except:
        pass

@with_setup(teardown=_remove_file)
def test_read_back():
    simple = np.arange(16*16).reshape((16,16))
    simple = simple.astype(np.uint8)
    imsave(_filename, simple)
    back = imread(_filename)
    assert np.all(simple == back)

@raises(RuntimeError)
def test_error():
    imread('imread/tests/data/error.tif')

def test_monochrome():
    mono = imread('imread/tests/data/mono.tif')
    assert mono.shape == (8,8)
    z = np.zeros((8,8),np.uint8)
    z.flat[::3] = 1
    assert np.all(z == mono)

