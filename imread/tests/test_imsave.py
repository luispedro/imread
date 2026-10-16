import pytest
from imread import imread, imsave, imsave_multi, imread_multi
import numpy as np

def test_non_existing():
    with pytest.raises(Exception):
        # in 0.2.5 this led to a hard crash!
        arr = np.arange(64,dtype=np.uint8).reshape((8,8))
        imsave('test-me.png', arr, 'some format which does not exist')


def test_bad_args():
    with pytest.raises(TypeError):
        arr = np.arange(64,dtype=np.uint8).reshape((8,8))
        imsave('test-me.png', arr, arr)


def test_save_float():
    with pytest.raises(TypeError):
        im = (np.arange(64*64).reshape((64,64)) % 32 ) * 2.
        imsave('test.jpeg', im)


@pytest.mark.parametrize('ext', ['png', 'tiff'])
def test_save_big_endian(ext):
    arr = np.arange(16, dtype='>u2').reshape((4,4))
    imsave('test-big-endian.' + ext, arr)
    assert np.all(imread('test-big-endian.' + ext) == arr)


def test_save_multi_big_endian():
    arrs = [(np.arange(16).reshape((4,4)) + i).astype('>u2') for i in range(3)]
    imsave_multi('test-big-endian.tiff', arrs)
    for a, b in zip(imread_multi('test-big-endian.tiff'), arrs):
        assert np.all(a == b)
