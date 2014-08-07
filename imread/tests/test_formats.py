import imread
def test_detect_format():
    assert imread.detect_format('imread/tests/data/GOOD.PNG') == 'png'
    assert imread.detect_format('imread/tests/data/ghsystem_flame.jpg') == 'jpeg'
    data = open('imread/tests/data/ghsystem_flame.jpg', 'rb').read(16)
    assert imread.detect_format(data, 1) == 'jpeg'

    assert imread.detect_format(b'\xff\xd8\xff\0', 1) == 'jpeg'
