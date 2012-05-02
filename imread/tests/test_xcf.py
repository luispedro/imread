from imread import imread

def test_xcf():
    im = imread('imread/tests/data/diag.xcf')
    assert im.shape == (8, 8, 3)
    assert im.max(2).diagonal().sum() == 0
