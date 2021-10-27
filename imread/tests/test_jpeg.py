import pytest
import numpy as np
from imread import imread, imsave
from . import file_path
import glob

_filename = 'imread_testing_file.jpg'

@pytest.fixture(autouse=True)
def _remove_files():
    yield
    from os import unlink
    from glob import glob
    filelist = glob("*.jpg")
    for f in filelist:
        try:
            unlink(f)
        except:
            pass

def test_jpeg():
    f = np.arange(64*16).reshape((64,16))
    f %= 16
    f = f.astype(np.uint8)
    imsave(_filename, f, 'jpeg')
    g = imread(_filename).squeeze()
    assert np.mean(np.abs(f.astype(float)-g)) < 1.


def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.jpg'))

def test_error_noent():
    with pytest.raises(OSError):
        imread(file_path('this-file-does-not-exist.jpeg'))


def test_quality():
    def pixel_diff(a):
        return np.mean(np.abs(a.astype(float) - data))

    data = np.arange(256*256*3)
    data %= 51
    data = data.reshape((256,256,3))
    data = data.astype(np.uint8)
    imsave('imread_def.jpg', data)
    imsave('imread_def91.jpg', data, opts={'jpeg:quality': 91} )
    readback    = imread('imread_def.jpg')
    readback91  = imread('imread_def91.jpg')
    assert pixel_diff(readback91) < pixel_diff(readback)
