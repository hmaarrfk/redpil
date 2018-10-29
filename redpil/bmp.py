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

gray_color_table_uint8 = np.arange(256, dtype='<u1')
gray_color_table_uint8 = np.stack([gray_color_table_uint8,
                                   gray_color_table_uint8,
                                   gray_color_table_uint8,
                                   np.full_like(gray_color_table_uint8,
                                                fill_value=0, dtype='<u1')],
                                  axis=1)

gray_color_table_bool = np.asarray([0, 255], dtype='<u1')
gray_color_table_bool = np.stack([gray_color_table_bool,
                                  gray_color_table_bool,
                                  gray_color_table_bool,
                                  np.full_like(gray_color_table_bool,
                                               fill_value=0, dtype='<u1')],
                                 axis=1)
def imwrite(filename, image):
    image = np.atleast_2d(image)
    if image.ndim > 2:
        raise NotImplementedError('Only monochrome images are supported.')

    if image.dtype == np.uint8:
        color_table = gray_color_table_uint8
        packed_image = image
    elif image.dtype == np.bool:
        color_table = gray_color_table_bool
        packed_image = np.packbits(image, axis=1)
    else:
        raise NotImplementedError('Only uint8 and bool images are supported.')

    header = np.zeros(1, dtype=header_t)
    info_header = np.empty(1, dtype=info_header_t)

    header['signature'] = 'BM'.encode()

    if image.dtype == np.bool:
        bits_per_pixel = 1
    else:
        bits_per_pixel = image.itemsize * 8
    # Not correct for color images
    # BMP wants images to be padded to a multiple of 4
    row_size = (bits_per_pixel * image.shape[1] + 31) // 32 * 4
    image_size = row_size * image.shape[0]

    header['file_offset_to_pixelarray'] = (header.nbytes +
                                           info_header.nbytes +
                                           color_table.nbytes)

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
        f.write(color_table)
        if row_size == packed_image.shape[1]:
            # Small optimization when the image is a multiple of 4 bytes
            # it actually avoids a full memory copy, so it is quite useful
            f.write(np.ascontiguousarray(packed_image).data)
        else:
            # Now slice just the part of the image that we actually write to.
            data = np.empty((image.shape[0], row_size), dtype=np.uint8)

            data[:packed_image.shape[0],
                 :packed_image.shape[1]] = packed_image
            f.write(data.data)


def imread(filename):
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

        compression = compression_types[info_header['compression'][0]]
        if compression != 'BI_RGB':
            raise NotImplementedError(
                "We only handle images with compression format BI_RGB. "
                "Got compression format {}.".format(compression))
        bits_per_pixel = info_header['bits_per_pixel'][0]
        if bits_per_pixel not in [1, 8]:
            raise NotImplementedError(
                "We only support images with 1 or 8 bits per pixel. Got "
                "{} bits per pixel.".format(bits_per_pixel)
            )

        color_table_max_shape = int(header['file_offset_to_pixelarray'][0] -
                                    header.nbytes - info_header.nbytes)
        color_table_count = min(color_table_max_shape, 2 ** bits_per_pixel * 4)
        if color_table_count:
            color_table = np.fromfile(
                f, dtype='<u1', count=color_table_count)
            color_table = color_table.reshape(-1, 4)
        else:
            color_table = None

        row_size = (bits_per_pixel * shape[1] + 31) // 32 * 4
        image_size = row_size * shape[0]

        f.seek(header['file_offset_to_pixelarray'][0])
        image = np.fromfile(f, dtype='<u1',
                            count=image_size).reshape(-1, row_size)
        # BMPs are saved typically as the last row first.
        # Except if the image height is negative
        if info_header['image_height'] > 0:
            image = image[::-1]

        # do a color table lookup
        if compression == 'BI_RGB':

            color_table = color_table[..., :3]
            if bits_per_pixel == 1:
                image = np.unpackbits(image, axis=1)
                gray_color_table = gray_color_table_bool
            else:
                gray_color_table = gray_color_table_uint8
            # These images are padded, make sure you slice them after unpacking
            # them
            image = image[:shape[0], :shape[1]]

            # Compress the color table if applicable
            if np.all(color_table[:, 0:1] == color_table[:, 1:3]):
                color_table = color_table[:, 0]
                gray_color_table = gray_color_table[:, 0]
            else:
                # Color table is provided in BGR, not RGB
                color_table = color_table[:, ::-1]

            if bits_per_pixel < 8 or not np.array_equal(color_table, gray_color_table):
                image = color_table[image]
        else:
            raise NotImplementedError('How did you get here')

    return image
