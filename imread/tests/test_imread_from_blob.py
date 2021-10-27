import pytest
import imread
from imread.imread import imread_from_blob
import numpy as np

test_imread_from_blob_data = [
    ('good.png', 'png'),
    ('good.png', None),
    ('GOOD.PNG', 'png'),
    ('mono.tif', 'tif'),
    ('mono.tif', 'tiff'),
    ('py-installer-indexed.bmp', 'bmp'),
]

@pytest.mark.parametrize("filename,formatstr", test_imread_from_blob_data)
def test_imread_from_blob(filename, formatstr):
    from os import path
    filename = path.join(
                    path.dirname(__file__),
                    'data',
                    filename)
    fromfile= imread.imread(filename)
    fromblob = imread_from_blob(open(filename, 'rb').read(), formatstr)
    assert np.all(fromblob == fromfile)


