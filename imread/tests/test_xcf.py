import sys 

from imread import imread

if not sys.platform.startswith('win'):
    # there is no native xcf2png utility for Windows
    def test_xcf():
        im = imread('imread/tests/data/diag.xcf')
        assert im.shape == (8, 8, 3)
        assert im.max(2).diagonal().sum() == 0
