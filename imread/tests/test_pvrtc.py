from nose.tools import raises
from imread import imread

@raises(RuntimeError)
def test_error():
    imread('imread/tests/data/error.webp')

def test_AI88():
    imread('imread/tests/data/pvrsamples/AI88.pvr')

def test_I8():
    imread('imread/tests/data/pvrsamples/I88.pvr')

def test_RGB565():
    imread('imread/tests/data/pvrsamples/RGB565.pvr')

def test_RGB888():
    imread('imread/tests/data/pvrsamples/RGB888.pvr')

def test_RGBA4444():
    imread('imread/tests/data/pvrsamples/RGBA4444.pvr')

def test_RGBA5551():
    imread('imread/tests/data/pvrsamples/RGBA5551.pvr')

def test_RGBA8888():
    imread('imread/tests/data/pvrsamples/RGBA8888.pvr')

def test_apple_2bpp():
    imread('imread/tests/data/pvrsamples/apple_2bpp.pvr')

def test_apple_4bpp():
    imread('imread/tests/data/pvrsamples/apple_4bpp.pvr')

def test_pvrtc2bpp():
    imread('imread/tests/data/pvrsamples/pvrtc2bpp.pvr')

def test_pvrtc4bpp():
    imread('imread/tests/data/pvrsamples/pvrtc4bpp.pvr')

# pngreference.png ...?


    
