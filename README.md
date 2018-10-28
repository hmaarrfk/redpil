# redpil


[![pypi](https://img.shields.io/pypi/v/redpil.svg)](https://pypi.python.org/pypi/redpil)
[![Travis](https://img.shields.io/travis/hmaarrfk/redpil.svg)](https://travis-ci.org/hmaarrfk/redpil)
[![Docs](https://readthedocs.org/projects/redpil/badge/?version=latest)](https://redpil.readthedocs.io/en/latest/?badge=latest)


Join the wonderland of python, and decode all your images in a numpy compatible way.

Pillow's memory system isn't compatible with numpy. Meaning that everytime you read or write images, they get copied to a Pillow array, then again to a numpy array.

For large images, this is a serious bottleneck.


* Documentation: https://redpil.readthedocs.io.


Features
--------

| Fileformat | Version | Write | Read | In scope | Reference                       |
|:-----------|:--------|:------|:-----|:---------|:--------------------------------|
| BMP        | Mono 8 bit only | Yes    | No   | Yes      | [Wikipedia](https://en.wikipedia.org/wiki/BMP_file_format) |
| JPEG       |         | No    | No   | Yes      |                                 |
| JPEG2000   |         | No    | No   | Yes      |                                 |
| GIF        |         | No    | No   | Yes      |                                 |
| PNG        |         | No    | No   | Yes      |                                 |
| SVG        |         | No    | No   | Maybe    |                                 |
| TIFF       |         | No    | No   | Maybe    |                                 |
