from imread import imread

def test_with_dot():
    f = imread('./imread/tests/data/good.png')
    assert f.shape == (2,2)
