from nose.tools import raises
from imread import imread


@raises(RuntimeError)
def test_error():
    imread('imread/tests/data/error.unknown')
