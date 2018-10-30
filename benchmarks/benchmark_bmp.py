import numpy as np
from PIL import Image
from tempfile import mkdtemp
from redpil.bmp import imwrite, imread
from pathlib import Path
import imageio
import shutil

class BMPSuite8bpp:
    params = ([(128, 128), (1024, 1024),
               (2048, 4096)], # (2**5 * 1024, 2 ** 5 *1024)],
              ['redpil', 'pillow', 'imageio'])
    param_names = ['shape', 'mode']

    def setup(self, shape, mode):
        self.img = np.random.randint(255, size=shape, dtype=np.uint8)
        self.tmpdir = Path(mkdtemp())
        self.filename = self.tmpdir / 'saved.bmp'
        imwrite(self.filename, self.img)

    def time_save(self, shape, mode):
        if mode == 'pillow':
            p = Image.fromarray(self.img)
            filename = self.tmpdir / 'pillow.bmp'
            p.save(filename)
        elif mode == 'imageio':
            filename = self.tmpdir / 'imageio.bmp'
            imageio.imwrite(filename, self.img)
        else:
            filename = self.tmpdir / 'redpil.bmp'
            imwrite(filename, self.img)

    def time_load(self, shape, mode):
        if mode == 'pillow':
            try:
                img = np.asarray(Image.open(self.filename).convert('L'))
            except Image.DecompressionBombError:
                pass
        elif mode == 'imageio':
            img = imageio.imread(self.filename)
        else:
            img = imread(self.filename)
        assert np.array_equal(img, self.img)

    def teardown(self, shape, mode):
        shutil.rmtree(self.tmpdir)



class BMPSuite24bpp:
    params = ([(32, 128, 3), (256, 1024, 3),
               (2048, 1024, 3)], #, (2**5 * 1024, 2 ** 3 *1024, 3)],
              ['redpil', 'pillow', 'imageio'])
    param_names = ['shape', 'mode']

    def setup(self, shape, mode):
        self.img = np.random.randint(255, size=shape, dtype=np.uint8)
        self.tmpdir = Path(mkdtemp())
        self.filename = self.tmpdir / 'saved.bmp'
        imwrite(self.filename, self.img)

    def time_save(self, shape, mode):
        if mode == 'pillow':
            p = Image.fromarray(self.img)
            filename = self.tmpdir / 'pillow.bmp'
            p.save(filename)
        elif mode == 'imageio':
            filename = self.tmpdir / 'imageio.bmp'
            imageio.imwrite(filename, self.img)
        else:
            filename = self.tmpdir / 'redpil.bmp'
            imwrite(filename, self.img)

    def time_load(self, shape, mode):
        if mode == 'pillow':
            try:
                img = np.asarray(Image.open(self.filename).convert('RGB'))
            except Image.DecompressionBombError:
                pass
        elif mode == 'imageio':
            img = imageio.imread(self.filename)
        else:
            img = imread(self.filename)

        assert np.array_equal(img, self.img)


    def teardown(self, shape, mode):
        shutil.rmtree(self.tmpdir)


class BMPSuite32bpp:
    params = ([(32, 128, 4), (256, 1024, 4),
               (2048, 1024, 4)], #, (2**5 * 1024, 2 ** 3 *1024, 4)],
              ['redpil', 'pillow', 'imageio'])
    param_names = ['shape', 'mode']

    def setup(self, shape, mode):
        self.img = np.random.randint(255, size=shape, dtype=np.uint8)
        self.tmpdir = Path(mkdtemp())
        self.filename = self.tmpdir / 'saved.bmp'
        imwrite(self.filename, self.img)
        self.filename_pillow = self.tmpdir / 'saved_pillow.bmp'
        # Pillow needs BGRA
        imwrite(self.filename_pillow, self.img, write_order='BGRA')


    def time_save(self, shape, mode):
        if mode == 'pillow':
            p = Image.fromarray(self.img)
            filename = self.tmpdir / 'pillow.bmp'
            p.save(filename)
        elif mode == 'imageio':
            filename = self.tmpdir / 'imageio.bmp'
            imageio.imwrite(filename, self.img)
        else:
            filename = self.tmpdir / 'redpil.bmp'
            imwrite(filename, self.img)

    def time_load(self, shape, mode):
        if mode == 'pillow':
            try:
                img = np.asarray(
                    Image.open(self.filename_pillow).convert('RGBA'))
            except Image.DecompressionBombError:
                pass
        elif mode == 'imageio':
            img = imread(self.filename)
        else:
            img = imread(self.filename)
        assert np.array_equal(img, self.img)

    def teardown(self, shape, mode):
        shutil.rmtree(self.tmpdir)
