from redpil.bmp import imread
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from numpy.testing import assert_array_equal


parametrize = pytest.mark.parametrize

good_folder = Path(__file__).parent / 'bmpsuite' / 'g'

all_good_files = ['pal1bg.bmp', 'pal4.bmp', 'pal8-0.bmp', 'pal8nonsquare.bmp',
                  'pal8topdown.bmp', 'pal8w124.bmp', 'rgb16-565.bmp',
                  'rgb16.bmp', 'rgb32bf.bmp', 'pal1.bmp', 'pal4gs.bmp',
                  'pal8.bmp', 'pal8os2.bmp', 'pal8v4.bmp', 'pal8w125.bmp',
                  'rgb16-565pal.bmp', 'rgb24.bmp', 'rgb32bfdef.bmp',
                  'pal1wb.bmp', 'pal4rle.bmp', 'pal8gs.bmp', 'pal8rle.bmp',
                  'pal8v5.bmp', 'pal8w126.bmp', 'rgb16bfdef.bmp',
                  'rgb24pal.bmp', 'rgb32.bmp']
all_good_files.sort()

passing_pillow_files = all_good_files.copy()

# Pillow OSError: Unsupported BMP compression (2)
passing_pillow_files.remove('pal4rle.bmp')
# Pillow OSError: Unsupported BMP bitfields layout
passing_pillow_files.remove('rgb32bf.bmp')
# Pillow OSError: Unsupported BMP compression (1)
passing_pillow_files.remove('pal8rle.bmp')

passing_files = ['pal1.bmp', 'pal1bg.bmp', 'pal1wb.bmp',
                 'pal4.bmp', 'pal4gs.bmp',
                 'pal8os2.bmp',
                 'pal8-0.bmp', 'pal8.bmp', 'pal8gs.bmp',
                 'pal8v4.bmp', 'pal8v5.bmp',
                 'pal8nonsquare.bmp', 'pal8topdown.bmp', 'pal8w124.bmp',
                 'pal8w125.bmp', 'pal8w126.bmp',
                 'rgb16.bmp',
                 'rgb16-565.bmp', 'rgb16-565pal.bmp', 'rgb16bfdef.bmp',
                 'rgb24.bmp', 'rgb24pal.bmp',
                 'rgb32.bmp']


non_passing_files = ['pal4rle.bmp',  'pal8rle.bmp', 'rgb32bf.bmp'
                     'rgb32bfdef.bmp']

# @pytest.mark.xfail
@parametrize('filename', passing_files)
def test_test(filename):
    imagepath = good_folder / filename
    img = imread(imagepath)
    img_pil = Image.open(imagepath)
    if img.ndim == 3:
        if img.shape[2] == 3:
            img_pil = img_pil.convert('RGB')
        elif img.shape[2] == 4:
            img_pil = img_pil.convert('RGBA')
    else:
        img_pil = img_pil.convert('L')

    img_pil = np.asarray(img_pil)
    assert_array_equal(img, img_pil)
