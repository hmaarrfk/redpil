# redpil

[![pypi](https://img.shields.io/pypi/v/redpil.svg)](https://pypi.python.org/pypi/redpil)
[![Travis](https://img.shields.io/travis/hmaarrfk/redpil.svg)](https://travis-ci.org/hmaarrfk/redpil)
[![Docs](https://readthedocs.org/projects/redpil/badge/?version=latest)](https://redpil.readthedocs.io/en/latest/?badge=latest)


Join the wonderland of python, and decode all your images in a numpy compatible
way.

Pillow is a great library for image manipulation. However, many operations fall
outside what Pillow can do. As such, many scientific applications require the
image to be available as a numpy array. However, Pillow's memory system
is largely incompatible with numpy's. [imageio](
https://github.com/imageio/imageio) has created an efficient bridge between
numpy and Pillow (see benchmarks below). Unfortunately, Pillow's multitude of
options remain confusing it is challenging to understand how they all operate
together. Furthermore, the code base is rather old, written in C, meaning that
it is difficult to extend the functionality of existing decoders.

For large images, having to understand the details of both Pillow and numpy is a serious bottleneck.
The goal of the library it to read and write images in a manner natural to numpy
users. Images are presented as the values they hold (not indices in a color
table) allowing for direct data analysis.

As much as possible, the library is written in python allowing for new decoding
algorithms to be played around with.


## Bitmap images
Generally, this library will not load memory in a C-contiguous array. Rather
the memory order will mostly match what was saved on disk.

Bitmap images will be stored in an order similar to how they arranged in
RAM.

## Supported file formats

Reading BMP is almost fully supported. Writing is still limited.

* BMP: 1, 4, or 8bit per pixel. [Wikipedia](https://en.wikipedia.org/wiki/BMP_file_format)

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
================ ============ ============ ============
--                                mode                 
---------------- --------------------------------------
     shape          redpil       pillow      imageio   
================ ============ ============ ============
   (128, 128)      93.4±1μs     254±30μs     369±20μs  
  (1024, 1024)     720±30μs     936±50μs    1.60±0.3ms
  (2048, 4096)    5.25±0.7ms   5.20±0.1ms    10.4±2ms  
 (32768, 32768)    480±10ms     489±5ms     1.34±0.09s
================ ============ ============ ============
```

Reading image
```
================ ============= ============ =============
--                                 mode                  
---------------- ----------------------------------------
     shape           redpil       pillow       imageio   
================ ============= ============ =============
   (128, 128)       131±5μs      293±10μs      130±2μs   
  (1024, 1024)      194±10μs    1.03±0.1ms     192±5μs   
  (2048, 4096)    1.69±0.05ms    8.55±1ms    1.67±0.03ms
 (32768, 32768)     350±3ms      230±5μs       354±10ms  
================ ============= ============ =============
```

Note, Pillow refuses to read the 1GB image because it thinks it is a fork bomb.

#### Patched up imageio

As it can be seen, the team at imageio/scikit-image are much better at reading
the pillow documentation and understanding how to use it effectively. Their
reading speeds actually match the reading speeds of redpil, even though they
use pillow as a backend. They even handle what pillow thinks is a forkbomb.

Through writing this module, two bugs were found in imageio that affect
the speed of saving images [imageio PR #398](
https://github.com/imageio/imageio/pull/398), and how images were being read
[imageio PR #399](
https://github.com/imageio/imageio/pull/399#issuecomment-433992314)

With PR 398, the saving speed of imageio+pillow now matches that of redpil.
Note I'm always using the computer when running benchmarks, so take the exact
numbers with a grain of salt.

Saving
```
================ ============ ============ ============
--                                mode                 
---------------- --------------------------------------
     shape          redpil       pillow      imageio   
================ ============ ============ ============
   (128, 128)      98.3±4μs     245±7μs      350±4μs   
  (1024, 1024)     714±20μs     921±30μs     997±20μs  
  (2048, 4096)    4.83±0.3ms   5.30±0.4ms   5.26±0.2ms
 (32768, 32768)    520±40ms     516±30ms     489±9ms   
================ ============ ============ ============
```

Reading
```
================ ============= ============ =============
--                                 mode                  
---------------- ----------------------------------------
     shape           redpil       pillow       imageio   
================ ============= ============ =============
   (128, 128)      129±0.7μs     284±2μs      129±0.7μs  
  (1024, 1024)      191±2μs     1.12±0.1ms    190±0.9μs  
  (2048, 4096)    1.62±0.03ms    8.88±1ms    1.63±0.02ms
 (32768, 32768)     357±9ms      223±4μs       361±8ms   
================ ============= ============ =============
```
