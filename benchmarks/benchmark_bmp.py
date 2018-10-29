import numpy as np
from PIL import Image
from tempfile import mkdtemp
from redpil.bmp import imwrite, imread
from pathlib import Path
import imageio
import shutil

class BMPSuite:
    params = ([(128, 128), (1024, 1024),
               (2048, 4096), (2**5 * 1024, 2 ** 5 *1024)],
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
                np.asarray(Image.open(self.filename))
            except Image.DecompressionBombError:
                pass
        elif mode == 'imageio':
            imread(self.filename)
        else:
            imread(self.filename)

    def teardown(self, shape, mode):
        shutil.rmtree(self.tmpdir)
