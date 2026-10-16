import pytest
from imread import imread
from . import file_path

def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.unknown'))

@pytest.mark.parametrize('opts,exc', [
    ({1: 2}, TypeError),
    ({'x': [1]}, TypeError),
    ({'\udcff': 1}, UnicodeEncodeError),
    ({'x': '\udcff'}, UnicodeEncodeError),
    ({'x': 2**100}, OverflowError),
])
def test_bad_opts(opts, exc):
    with pytest.raises(exc):
        imread(file_path('good.png'), opts=opts)
