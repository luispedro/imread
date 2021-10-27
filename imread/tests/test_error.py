import pytest
from imread import imread
from . import file_path

def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.unknown'))
