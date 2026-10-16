import pytest
import numpy as np
from imread import imread, imsave, imread_multi, imsave_multi
from . import file_path

_filename = 'imread_testing_file.tiff'

def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.tif'))


def test_read_back():
    simple = np.arange(16*16).reshape((16,16))
    simple = simple.astype(np.uint8)
    imsave(_filename, simple)
    back = imread(_filename)
    assert np.all(simple == back)

def test_read_back_16():
    np.random.seed(21)
    simple = np.random.random_sample((128,128))
    simple *= 8192
    simple = simple.astype(np.uint16)
    imsave(_filename, simple)
    back = imread(_filename)
    assert np.all(simple == back)

def test_monochrome():
    mono = imread(file_path('mono.tif'))
    assert mono.shape == (8,8)
    z = np.zeros((8,8),np.uint8)
    z.flat[::3] = 1
    assert np.all(z == mono)


def test_multi():
    assert len(imread_multi(file_path('stack.tiff'))) == 2

def test_read_back_with_metadata():
    simple = np.arange(16*16).reshape((16,16))
    simple = simple.astype(np.uint8)
    meta = b'123qwe'
    imsave(_filename, simple, metadata=meta)
    back,meta_read = imread(_filename, return_metadata=True)
    assert np.all(simple == back)
    assert meta == meta_read


def test_read_back_colour():
    im = np.arange(256).astype(np.uint8).reshape((32,-1))
    im = np.dstack([im, im*0, 255-im])
    imsave(_filename, im)
    im2 = imread(_filename)
    assert im.shape == im2.shape
    assert np.all(im == im2)

def test_read_back_colour_16bit():
    im = np.random.random((16,8,3)) * 65535.0
    im = im.astype(np.uint16)
    imsave(_filename, im)
    im2 = imread(_filename)
    assert im.shape == im2.shape
    assert np.all(im == im2)

def test_horizontal_predictor():
    im = imread(file_path('arange512_16bit.png'))
    im2 = im.copy()
    imsave(_filename, im, opts={'tiff:horizontal-predictor': True})
    assert np.all(im == im2)

    im3 = imread(_filename)
    assert np.all(im == im3)

def _read_resolution(fname):
    '''Returns {tag: value} for the XResolution (282) and YResolution (283)
    tags in the first directory of a TIFF file'''
    import struct
    with open(fname, 'rb') as f:
        data = f.read()
    bo = '<' if data[:2] == b'II' else '>'
    ifd, = struct.unpack(bo + 'I', data[4:8])
    n, = struct.unpack(bo + 'H', data[ifd:ifd+2])
    res = {}
    for i in range(n):
        tag, _, _, offset = struct.unpack(bo + 'HHII', data[ifd+2+12*i:ifd+14+12*i])
        if tag in (282, 283):
            num, den = struct.unpack(bo + 'II', data[offset:offset+8])
            res[tag] = num / den
    return res

@pytest.mark.parametrize('opts,expected', [
    ({}, {}),
    # If only one is given, it is used for both
    ({'tiff:XResolution': 100}, {282: 100., 283: 100.}),
    ({'tiff:YResolution': 50}, {282: 50., 283: 50.}),
    ({'tiff:XResolution': 100, 'tiff:YResolution': 50}, {282: 100., 283: 50.}),
    ({'tiff:XResolution': 72.5, 'tiff:YResolution': 36.25}, {282: 72.5, 283: 36.25}),
    ])
def test_resolution(opts, expected):
    im = np.zeros((4, 4), np.uint8)
    imsave(_filename, im, opts=opts)
    assert _read_resolution(_filename) == expected

def test_resolution_bad_value():
    im = np.zeros((4, 4), np.uint8)
    with pytest.raises(RuntimeError):
        imsave(_filename, im, opts={'tiff:YResolution': 'high'})

def test_imsave_multi():
    im = imread(file_path('arange512_16bit.png'))
    im2 = im[::4, ::4]
    ims = [im, im2]
    imsave_multi(_filename, ims)
    ims2 = imread_multi(_filename)
    assert len(ims) == len(ims2)
    for a,b in zip(ims, ims2):
        assert np.all(a == b)
