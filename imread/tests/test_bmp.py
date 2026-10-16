import struct
import numpy as np
import pytest
from imread import imread
from . import file_path

def test_read():
    im = imread(file_path('star1.bmp'))
    assert np.any(im)
    assert im.shape == (128, 128, 3)

def test_indexed():
    im = imread(file_path('py-installer-indexed.bmp'))
    assert np.any(im)
    assert im.shape == (352, 162, 3)
    assert np.any(im[:,:,0])
    assert np.any(im[:,:,1])
    assert np.any(im[:,:,2])


def _write_bmp(fname, rows, width, height, bpp, compression=0, header_size=40,
               masks=(), palette=None, n_colours=None):
    '''Write a BMP file

    rows: list of byte strings, in file order (without padding)
    masks: colour masks to store (after the header, or inside it if
    header_size > 40)
    palette: list of (r, g, b) tuples
    '''
    stride = (width * bpp + 31) // 32 * 4
    data = b''.join(r.ljust(stride, b'\0') for r in rows)
    if palette is None:
        palette = []
    if n_colours is None:
        n_colours = len(palette)
    table = b''.join(bytes([b, g, r, 0]) for r, g, b in palette)
    header = struct.pack('<IiiHHIIiiII',
                header_size, width, height, 1, bpp, compression,
                len(data), 2835, 2835, n_colours, 0)
    masks = struct.pack('<{}I'.format(len(masks)), *masks)
    if header_size > 40:
        header = (header + masks).ljust(header_size, b'\0')
    else:
        header += masks
    offset = 14 + len(header) + len(table)
    with open(fname, 'wb') as out:
        out.write(b'BM' + struct.pack('<IHHI', offset + len(data), 0, 0, offset))
        out.write(header)
        out.write(table)
        out.write(data)


def _rgb_image(h=3, w=5):
    return np.arange(h * w * 3, dtype=np.uint8).reshape((h, w, 3)) * 5

def _rows_24(im):
    return [bytes(row[:, ::-1].ravel()) for row in im]

def test_24bit_bottom_up():
    im = _rgb_image()
    _write_bmp('x.bmp', _rows_24(im[::-1]), im.shape[1], im.shape[0], 24)
    assert np.array_equal(imread('x.bmp'), im)

def test_top_down():
    im = _rgb_image()
    _write_bmp('x.bmp', _rows_24(im), im.shape[1], -im.shape[0], 24)
    assert np.array_equal(imread('x.bmp'), im)

def test_missing_last_row_padding():
    im = _rgb_image()
    _write_bmp('x.bmp', _rows_24(im[::-1]), im.shape[1], im.shape[0], 24)
    data = open('x.bmp', 'rb').read()
    # Each row has 15 bytes of data and 1 byte of padding
    open('x.bmp', 'wb').write(data[:-1])
    assert np.array_equal(imread('x.bmp'), im)

def test_truncated():
    im = _rgb_image()
    _write_bmp('x.bmp', _rows_24(im[::-1]), im.shape[1], im.shape[0], 24)
    data = open('x.bmp', 'rb').read()
    open('x.bmp', 'wb').write(data[:-8])
    with pytest.raises(RuntimeError):
        imread('x.bmp')

def test_zero_height():
    _write_bmp('x.bmp', [], 4, 0, 24)
    with pytest.raises(RuntimeError):
        imread('x.bmp')


_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30),
            (1, 2, 3), (4, 5, 6), (7, 8, 9), (100, 110, 120),
            (200, 0, 0), (0, 200, 0), (0, 0, 200), (50, 60, 70),
            (11, 12, 13), (14, 15, 16), (17, 18, 19), (240, 250, 255)]

def _pack_indices(indices, bpp):
    bits = ''.join(format(int(i), '0{}b'.format(bpp)) for i in indices)
    bits = bits.ljust((len(bits) + 7) // 8 * 8, '0')
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

@pytest.mark.parametrize('bpp', [1, 4, 8])
def test_paletted(bpp):
    h, w = 3, 13
    indices = np.arange(h * w).reshape((h, w)) % (2**bpp)
    palette = _PALETTE[:2**bpp] if bpp < 8 else (_PALETTE * 16)
    _write_bmp('x.bmp', [_pack_indices(row, bpp) for row in indices[::-1]],
               w, h, bpp, palette=palette)
    expected = np.array(palette, dtype=np.uint8)[indices]
    assert np.array_equal(imread('x.bmp'), expected)

@pytest.mark.parametrize('header_size', [40, 108, 124])
def test_paletted_header_size(header_size):
    indices = np.array([[0, 1, 2, 3]])
    _write_bmp('x.bmp', [bytes(indices[0].astype(np.uint8))], 4, 1, 8,
               header_size=header_size, palette=_PALETTE[:4])
    expected = np.array(_PALETTE[:4], dtype=np.uint8)[indices]
    assert np.array_equal(imread('x.bmp'), expected)

def test_huge_n_colours():
    _write_bmp('x.bmp', [_pack_indices([0, 1], 1)], 2, 1, 1,
               palette=_PALETTE[:2], n_colours=0x10000000)
    assert np.array_equal(imread('x.bmp'), [_PALETTE[:2]])

def test_index_outside_palette():
    _write_bmp('x.bmp', [bytes([0, 5])], 2, 1, 8, palette=_PALETTE[:2])
    with pytest.raises(RuntimeError):
        imread('x.bmp')


def _pack(values, fmt):
    return struct.pack('<{}{}'.format(len(values), fmt), *values)

def test_16bit_rgb555():
    pixels = [0x7c00, 0x03e0, 0x001f, 0x7fff, 0x0000, (16 << 10) | (8 << 5) | 1]
    _write_bmp('x.bmp', [_pack(pixels, 'H')], len(pixels), 1, 16)
    s = lambda v: (v * 255 + 15) // 31
    expected = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0],
                [s(16), s(8), s(1)]]
    im = imread('x.bmp')
    assert im.dtype == np.uint8
    assert np.array_equal(im, [expected])

def test_16bit_rgb565():
    pixels = [0xf800, 0x07e0, 0x001f, (16 << 11) | (32 << 5) | 1]
    _write_bmp('x.bmp', [_pack(pixels, 'H')], len(pixels), 1, 16,
               compression=3, masks=(0xf800, 0x07e0, 0x001f))
    expected = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
                [(16 * 255 + 15) // 31, (32 * 255 + 31) // 63, (255 + 15) // 31]]
    assert np.array_equal(imread('x.bmp'), [expected])

def test_32bit_rgb():
    im = _rgb_image()
    rows = [_pack([0xff000000 | (r << 16) | (g << 8) | b for r, g, b in row.astype(int)], 'I')
            for row in im[::-1]]
    _write_bmp('x.bmp', rows, im.shape[1], im.shape[0], 32)
    assert np.array_equal(imread('x.bmp'), im)

@pytest.mark.parametrize('header_size,compression', [(40, 6), (108, 3), (124, 3)])
def test_32bit_alpha(header_size, compression):
    im = np.arange(2 * 3 * 4, dtype=np.uint8).reshape((2, 3, 4)) * 7
    rows = [_pack([(a << 24) | (b << 16) | (g << 8) | r for r, g, b, a in row.astype(int)], 'I')
            for row in im[::-1]]
    _write_bmp('x.bmp', rows, im.shape[1], im.shape[0], 32,
               compression=compression, header_size=header_size,
               masks=(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
    assert np.array_equal(imread('x.bmp'), im)

def test_32bit_bitfields_no_alpha():
    im = _rgb_image()
    rows = [_pack([(r << 24) | (g << 16) | (b << 8) | 0xff for r, g, b in row.astype(int)], 'I')
            for row in im[::-1]]
    _write_bmp('x.bmp', rows, im.shape[1], im.shape[0], 32,
               compression=3, masks=(0xff000000, 0x00ff0000, 0x0000ff00))
    assert np.array_equal(imread('x.bmp'), im)

def test_noncontiguous_mask():
    _write_bmp('x.bmp', [_pack([0], 'I')], 1, 1, 32,
               compression=3, masks=(0x00ff00ff, 0x0000ff00, 0x000000ff))
    with pytest.raises(RuntimeError):
        imread('x.bmp')
