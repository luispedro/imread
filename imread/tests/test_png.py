import pytest
import numpy as np
from . import file_path
from imread import imread, imsave

_filename = 'imread_testing_file.png'

@pytest.fixture(autouse=True)
def _remove_file():
    from os import unlink
    try:
        unlink(_filename)
    except:
        pass


def test_png_raw():
    simple = np.arange(16*16).reshape((16,16))
    simple = simple.astype(np.uint8)
    imsave(_filename, simple, 'png')
    back = imread(_filename)
    assert np.all(simple == back)

def test_asym():
    simple = np.arange(16*16).reshape((32,8))
    simple = simple.astype(np.uint8)
    imsave(_filename, simple, 'png')
    back = imread(_filename)
    assert np.all(simple == back)

def test_random():
    np.random.seed(23)
    for i in range(8):
        simple = np.random.random_sample((64,64,3))
        simple *= 255
        simple = simple.astype(np.uint8)
        if i < 3:
            simple[:,:,i] = 0
        imsave(_filename, simple, 'png')
        back = imread(_filename)
        assert np.all(simple == back)


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
    f = imread(file_path('bit1.png'))
    assert f.dtype == np.bool_

def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.png'))

def test_regression():
    im = imread(file_path('palette_zero.png'))
    assert im.sum() == 0
    assert im.shape == (128, 64, 3)


def test_16bit():
    f = imread(file_path('arange512_16bit.png'))
    assert np.all(f.ravel() == np.arange(512))


def test_write_16bit():
    f = np.arange(100000, dtype=np.uint16)*1000
    f = f.reshape((100,-1))
    imsave(_filename, f)
    f2 = imread(_filename)
    assert np.all(f == f2)

def test_write_16bit_rgb():
    f = np.random.random((16,8,3)) * 65535.0
    f = f.astype(np.uint16)
    imsave(_filename, f)
    f2 = imread(_filename)
    assert np.all(f == f2)

def test_strip_alpha():
    w_alpha = imread(file_path('rgba.png'))
    wno_alpha = imread(file_path('rgba.png'), opts={'strip_alpha':True})
    assert np.all(w_alpha[:,:,:3] == wno_alpha)
