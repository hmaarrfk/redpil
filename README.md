# redpil

[![pypi](https://img.shields.io/pypi/v/redpil.svg)](https://pypi.python.org/pypi/redpil)
[![Travis](https://img.shields.io/travis/hmaarrfk/redpil.svg)](https://travis-ci.org/hmaarrfk/redpil)
[![Docs](https://readthedocs.org/projects/redpil/badge/?version=latest)](https://redpil.readthedocs.io/en/latest/?badge=latest)


Join the wonderland of python, and decode all your images in a numpy compatible way.

Pillow's memory system isn't compatible with numpy. Meaning that everytime you
read or write images, they get copied to a Pillow array, then again to a numpy
array.

For large images, this is a serious bottleneck.

* Documentation: https://redpil.readthedocs.io.

## Supported file formats

* BMP: 8bit mono grayscale only. [Wikipedia](https://en.wikipedia.org/wiki/BMP_file_format)

## Future file formats

* BMP: more coverage
* JPEG, JPEG2000
* GIF
* PNG
* SVG
* TIFF

## Benchmarks

I don't have a fancy benchmarking service like scikit-image or dask has, but
here are the benchmarks results compared to a PIL backend. This is running
on my SSD, a Samsung 960 Pro which claims it can write at 1.8GB/s. This is
pretty close to what `redpil` achieves.


### 8 bit BMP grayscale images

Saving images:
```
================ ============ =============
--                          mode           
---------------- --------------------------
     shape          pillow        redpil   
================ ============ =============
   (128, 128)      263±4μs       106±2μs   
  (1024, 1024)    1000±60μs      787±90μs  
  (2048, 4096)    5.15±0.2ms   4.62±0.07ms
 (32768, 32768)    510±10ms      470±3ms   
================ ============ =============
```

Reading image
```
================ =========== =============
--                          mode          
---------------- -------------------------
     shape          pillow       redpil   
================ =========== =============
   (128, 128)      306±7μs      120±2μs   
  (1024, 1024)    990±300μs     187±7μs   
  (2048, 4096)     3.53±1ms   1.39±0.04ms
 (32768, 32768)    242±7μs      369±4ms   
================ =========== =============
```
Note, Pillow refuses to read the 1GB image because it thinks it is a fork bomb.
