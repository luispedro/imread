import imread
from . import file_path

def test_detect_format():
    assert imread.detect_format(file_path('GOOD.PNG')) == 'png'
    assert imread.detect_format(file_path('ghsystem_flame.jpg')) == 'jpeg'
    data = open(file_path('ghsystem_flame.jpg'), 'rb').read(16)
    assert imread.detect_format(data, 1) == 'jpeg'

    assert imread.detect_format(b'\xff\xd8\xff\0', 1) == 'jpeg'


def test_supports_format():
    assert imread.supports_format("png")
    assert not imread.supports_format("xyz-format-which-does-not-exist")
