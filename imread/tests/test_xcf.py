import sys 
from imread import imread
from . import file_path

if not sys.platform.startswith('win'):
    # there is no native xcf2png utility for Windows
    def test_xcf():
        im = imread(file_path('diag.xcf'))
        assert im.shape == (8, 8, 3)
        assert im.max(2).diagonal().sum() == 0
