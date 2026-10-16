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



def _read_data(filename):
    from os import path
    with open(path.join(path.dirname(__file__), 'data', filename), 'rb') as f:
        return f.read()

@pytest.mark.parametrize("offset", [2**31 - 1, 2**32 - 1])
def test_bmp_offset_past_end(offset):
    import struct
    data = bytearray(_read_data('star1.bmp'))
    data[10:14] = struct.pack('<I', offset)
    with pytest.raises(RuntimeError):
        imread_from_blob(bytes(data), 'bmp')

@pytest.mark.parametrize("filename,formatstr", [
    ('mono.tif', 'tiff'),
    ('star1.bmp', 'bmp'),
])
def test_truncated_blob(filename, formatstr):
    data = _read_data(filename)
    for n in (0, 1, 8, len(data) // 2):
        with pytest.raises(RuntimeError):
            imread_from_blob(data[:n], formatstr)
