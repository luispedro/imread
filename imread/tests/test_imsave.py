import pytest
from imread import imsave
import numpy as np

def test_non_existing():
    with pytest.raises(Exception):
        # in 0.2.5 this led to a hard crash!
        arr = np.arange(64,dtype=np.uint8).reshape((8,8))
        imsave('/tmp/test-me.png', arr, 'some format which does not exist')


def test_bad_args():
    with pytest.raises(TypeError):
        arr = np.arange(64,dtype=np.uint8).reshape((8,8))
        imsave('/tmp/test-me.png', arr, arr)


def test_save_float():
    with pytest.raises(TypeError):
        im = (np.arange(64*64).reshape((64,64)) % 32 ) * 2.
        imsave('test.jpeg', im)
