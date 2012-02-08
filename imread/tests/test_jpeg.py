from nose.tools import with_setup, raises
import numpy as np
from imread import _imread
from imread import imread
_filename = 'imread_testing_file.jpg'

def _remove_file():
    from os import unlink
    try:
        unlink(_filename)
    except:
        pass

@with_setup(teardown=_remove_file)
def test_jpeg():
    f = np.arange(64*16).reshape((64,16))
    f %= 16
    f = f.astype(np.uint8)
    _imread.imsave(_filename, 'jpeg', f)
    g = _imread.imread(_filename).squeeze()
    assert np.mean(np.abs(f.astype(float)-g)) < 1.


@raises(RuntimeError)
def test_error():
    imread('imread/tests/data/error.jpg')

@raises(OSError)
def test_error():
    imread('imread/tests/data/this-file-does-not-exist.jpeg')
