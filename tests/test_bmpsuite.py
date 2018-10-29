from redpil.bmp import imread
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from numpy.testing import assert_array_equal


parametrize = pytest.mark.parametrize

good_folder = Path(__file__).parent / 'bmpsuite' / 'g'
all_good_images = [p
                   for p in sorted(good_folder.glob('*'))
                   if p.is_file() and p.suffix.lower() == '.bmp']

# Pillow OSError: Unsupported BMP compression (2)
all_good_images.remove(good_folder / 'pal4rle.bmp')
# Pillow OSError: Unsupported BMP bitfields layout
all_good_images.remove(good_folder / 'rgb32bf.bmp')
# Pillow OSError: Unsupported BMP compression (1)
all_good_images.remove(good_folder / 'pal8rle.bmp')

# @pytest.mark.xfail
@parametrize('imagepath', all_good_images)
def test_test(imagepath):
    img_pil = np.asarray(Image.open(imagepath))
    img = imread(imagepath)
    # assert_array_equal(img, img_pil)
