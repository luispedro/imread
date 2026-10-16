import pytest
from imread import imread
from . import file_path

def test_error():
    with pytest.raises(RuntimeError):
        imread(file_path('error.webp'))

def test_empty_blob():
    from imread import imread_from_blob
    with pytest.raises(RuntimeError):
        imread_from_blob(b'', 'webp')
