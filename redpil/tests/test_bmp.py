from redpil.bmp import imwrite, imread

import numpy as np
import pytest
from PIL import Image
from numpy.testing import assert_array_equal
from pathlib import Path

def test_failers(tmpdir):
    tmpfile = Path(tmpdir) / 'test.bmp'
    img = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        imwrite(tmpfile, img)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        imwrite(tmpfile, img)

@pytest.mark.parametrize('shape', [(4, 4), (7, 7), (21, 7)])
def test_zero_image(tmpdir, shape):
    tmpfile = Path(tmpdir) / 'test.bmp'

    img = np.random.randint(255, size=shape, dtype=np.uint8)
    imwrite(tmpfile, img)

    img_read = np.asarray(Image.open(tmpfile))
    assert_array_equal(img, img_read)


@pytest.mark.parametrize('shape', [(4, 4), (7, 7), (21, 7)])
def test_zero_image(tmpdir, shape):
    tmpfile = Path(tmpdir) / 'test.bmp'

    img = np.random.randint(255, size=shape, dtype=np.uint8)
    imwrite(tmpfile, img)

    img_read = imread(tmpfile)
    assert_array_equal(img, img_read)
