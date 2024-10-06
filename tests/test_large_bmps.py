from redpil.bmp import imwrite, imread
import pytest
import numpy as np

@pytest.mark.parametrize('shape', [
    (32, 16, 3),
    (32, 16),
    (256, 128, 3),
    (256, 128),
    (256, 512, 3),
    (256, 512),
    (1024, 512, 3),
    (1024, 512),
    (1024, 2048, 3),
    (1024, 2048),
    (4096, 2048, 3),
    (4096, 2048),
    (4096, 8192, 3),
    (4096, 8192),
])
def test_image_roundtrip(shape, tmp_path):
    image = np.random.randint(0, 256, shape, dtype=np.uint8)
    imwrite(tmp_path / 'image.bmp', image)
    image2 = imread(tmp_path / 'image.bmp')
    np.testing.assert_array_equal(image, image2)
