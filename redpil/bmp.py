import numpy as np
from io import SEEK_CUR

__all__ = ['imread', 'imwrite']

header_t = np.dtype([
    ('signature', '|S2'),
    ('filesize', '<u4'),
    ('reserved1', '<u2'),
    ('reserved2', '<u2'),
    ('file_offset_to_pixelarray', '<u4')
])

info_header_t = np.dtype([
    ('header_size', '<u4'),
    ('image_width', '<i4'),
    ('image_height', '<i4'),
    ('image_planes', '<u2'),
    ('bits_per_pixel', '<u2'),
    ('compression', '<u4'),
    ('image_size', '<u4'),
    ('x_pixels_per_meter', '<u4'),
    ('y_pixels_per_meter', '<u4'),
    ('colors_in_color_table', '<u4'),
    ('important_color_count', '<u4'),
])

compression_types = ['BI_RGB', 'BI_RLE8', 'BI_RLE4', 'BI_BITFIELDS', 'BI_JPEG',
                     'BI_PNG', 'BI_ALPHABITFIELDS', 'BI_CMYK', 'BI_CMYKRLE8'
                     'BI_CMYKRLE4']

gray_color_table = np.arange(256, dtype='<u1')
gray_color_table = np.stack([gray_color_table,
                             gray_color_table,
                             gray_color_table,
                             np.full_like(gray_color_table,
                                          fill_value=255)], axis=1)

def imwrite(filename, image):
    image = np.atleast_2d(image)
    if image.ndim > 2:
        raise NotImplementedError('Only monochrome images are supported.')

    if image.dtype != np.uint8:
        raise NotImplementedError('Only uint8 images are supported.')
    header = np.zeros(1, dtype=header_t)
    info_header = np.empty(1, dtype=info_header_t)

    header['signature'] = 'BM'.encode()

    bits_per_pixel = image.itemsize * 8
    # Not correct for color images
    # BMP wants images to be padded to a multiple of 4
    row_size = (bits_per_pixel * image.shape[1] + 31) // 32 * 4
    image_size = row_size * image.shape[0]


    header['file_offset_to_pixelarray'] = (header.nbytes +
                                           info_header.nbytes +
                                           gray_color_table.nbytes)

    header['filesize'] = (header['file_offset_to_pixelarray'] + image_size)

    info_header['header_size'] = info_header.nbytes
    info_header['image_width'] = image.shape[1]
    # A positive height states the the array is saved "bottom to top"
    # A negative height states that the array is saved "top to bottom"
    # Top to bottom has a larger chance of being contiguous in C memory
    info_header['image_height'] = -image.shape[0]
    info_header['image_planes'] = 1
    info_header['bits_per_pixel'] = bits_per_pixel
    info_header['compression'] = compression_types.index('BI_RGB')
    info_header['image_size'] = 0
    info_header['x_pixels_per_meter'] = 0
    info_header['y_pixels_per_meter'] = 0
    info_header['colors_in_color_table'] = 0
    info_header['important_color_count'] = 0

    with open(filename, 'bw+') as f:
        f.write(header)
        f.write(info_header)
        f.write(gray_color_table)
        if row_size == image.shape[1]:
            # Small optimization when the image is a multiple of 4 bytes
            # it actually avoids a full memory copy, so it is quite useful
            f.write(np.ascontiguousarray(image).data)
        else:
            # Now slice just the part of the image that we actually write to.
            data = np.empty((image.shape[0], row_size), dtype=np.uint8)
            data[:image.shape[0], :image.shape[1]] = image
            f.write(data.data)


def imread(filename):
    header = np.zeros(1, dtype=header_t)
    with open(filename, 'br') as f:
        header = np.fromfile(f, dtype=header_t, count=1)
        if header['signature'] != 'BM'.encode():
            raise ValueError('Provided file is not a bmp file.')
        header_size = np.fromfile(f, dtype=info_header_t['header_size'], count=1)
        if header_size != info_header_t.itemsize:
            raise NotImplementedError(
                'We only implement basic gray scale images.')
        f.seek(-info_header_t['header_size'].itemsize, SEEK_CUR)
        info_header = np.fromfile(f, dtype=info_header_t, count=1)

        shape = (int(abs(info_header['image_height'])), int(info_header['image_width']))
        if info_header['image_planes'] != 1:
            raise NotImplementedError(
                "We don't know how to handle more than 1 image plane. "
                "Got {} image planes.".format(info_header['image_planes']))

        if info_header['bits_per_pixel'] != 8:
            raise NotImplementedError(
                "We don't know how to handle images with more or less than 8 "
                "bits per pixel. Got {} bits per pixels".format(
                    info_header['bits_per_pixel']))

        if compression_types[info_header['compression'][0]] != 'BI_RGB':
            raise NotImplementedError(
                "We only handle images with compression format BI_RGB. "
                "Got compression format {}.".format(
                    compression_types(info_header['compression'])))

        color_table = np.fromfile(
            f, dtype='<u1', count=2 ** info_header['bits_per_pixel'][0] * 4)
        color_table = color_table.reshape(-1, 4)

        if not np.all(color_table == gray_color_table):
            raise NotImplementedError(
                'We only handle the case where the color table is that of a '
                'grayscale image.')

        row_size = (info_header['bits_per_pixel'][0] * shape[1] + 31) // 32 * 4
        image_size = row_size * shape[0]

        f.seek(header['file_offset_to_pixelarray'][0])
        image = np.fromfile(f, dtype='<u1',
                            count=image_size).reshape(-1, row_size)
        image = image[:shape[0], :shape[1]]
        # BMPs are saved typically as the last row first.
        # Except if the image height is negative
        if info_header['image_height'] > 0:
            image = image[::-1]

    return image
