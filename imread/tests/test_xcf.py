import sys 
import pytest
from imread import imread
from . import file_path

def has_xcf2png():
    # there is no native xcf2png utility for Windows
    if sys.platform.startswith('win'):
        return False
    try:
        from shutil import which
    except:
        from distutils.spawn import find_executable as which

    c = which('xcf2png')
    return (c is not None)


@pytest.mark.skipif(not has_xcf2png(), reason="do not have xcf2png utility")
def test_xcf():
    im = imread(file_path('diag.xcf'))
    assert im.shape == (8, 8, 3)
    assert im.max(2).diagonal().sum() == 0
