import numpy as np
from io import SEEK_CUR

__all__ = ['imread', 'imwrite']


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
    info_header = np.empty(1, dtype=bitmap_info_header_t)

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
        header_size = int(np.fromfile(f, dtype='<u4', count=1))

        if header_size not in header_sizes.values():
            raise ValueError(
                'Unsupported image format. Header size has a value of {}.'
                ''.format(header_size))
        f.seek(-bitmap_info_header_t['header_size'].itemsize, SEEK_CUR)
        info = np.fromfile(f, dtype=info_header_t_dict[header_size], count=1)
        info_header = np.zeros(1, dtype=bitmap_v5_header_t)
        for name in info.dtype.names:
            info_header[name] = info[name]

        shape = (int(abs(info_header['image_height'])), int(info_header['image_width']))
        if info_header['image_planes'] != 1:
            raise NotImplementedError(
                "We don't know how to handle more than 1 image plane. "
                "Got {} image planes.".format(info_header['image_planes']))

        compression = compression_types[info_header['compression'][0]]
        implemented_compressions = ['BI_RGB', 'BI_BITFIELDS']
        if compression not in implemented_compressions:
            raise NotImplementedError(
                "We only handle images with compression format {}. "
                "Got compression format {}.".format(
                    implemented_compressions, compression))
        bits_per_pixel = info_header['bits_per_pixel'][0]
        if bits_per_pixel not in _decoders.keys():
            raise NotImplementedError(
                "We only support images with one of {} bits per "
                "pixel. Got an image with {} bits per pixel.".format(
                    list(_decoders.keys()),
                    bits_per_pixel)
            )

        color_table_max_shape = int(header['file_offset_to_pixelarray'][0] -
                                    header.nbytes - info.nbytes)
        if info_header['colors_in_color_table'] != 0:
            color_table_max_shape = min(color_table_max_shape,
                                        int(info_header['colors_in_color_table']) * 4)
        color_table_count = min(color_table_max_shape, 2 ** bits_per_pixel * 4)

        # Bitfields doesn't use a color table
        if compression == 'BI_BITFIELDS':
            color_table_count = 0

        color_table = np.fromfile(f, dtype='<u1', count=color_table_count)

        if header_size == header_sizes['BITMAPCOREHEADER']:
            # bitmap core header color tables only contain 3 values, not 4
            color_table = color_table.reshape(-1, 3)
        else:
            color_table = color_table.reshape(-1, 4)

        # When color tables are used, alpha is ignored.
        color_table = color_table[..., :3]

        row_size = (bits_per_pixel * shape[1] + 31) // 32 * 4

        decoder = _decoders[bits_per_pixel]
        return decoder(f, header, info_header, color_table, shape, row_size)


def _compress_color_table(color_table):
    if np.all(color_table[:, 0:1] == color_table[:, 1:3]):
        return color_table[:, 0]
    else:
        # Color table is provided in BGR, not RGB
        return color_table[:, ::-1]


def _decode_32bpp(f, header, info_header, color_table, shape, row_size):
    compression = compression_types[info_header['compression'][0]]

    if compression == 'BI_BITFIELDS':
        bitfields = np.fromfile(f, dtype='<u4', count=3).tolist()
        right_shift = []
        precision = []
        for bitfield in bitfields:
            for i in range(32):
                if np.bitwise_and(bitfield, 0x1) == 1:
                    right_shift.append(i)
                    for j in range(i, 32):
                        bitfield = np.right_shift(bitfield, 1)
                        if np.bitwise_and(bitfield, 0x1) == 0:
                            precision.append(j - i + 1)
                            break
                    break
                bitfield = np.right_shift(bitfield, 1)

        bitfields_use_uint8 = (precision == [8, 8, 8] and
                               all(shift % 8 == 0 for shift in right_shift))
    else:
        bitfields = [0x0000FF00, 0x00FF0000, 0xFF000000]

    f.seek(int(header['file_offset_to_pixelarray']))

    image_size = row_size * shape[0]
    if (compression == 'BI_BITFIELDS' and
            not bitfields_use_uint8):
        image = np.fromfile(f, dtype='<u4',
                            count=image_size // 4).reshape(-1, row_size // 4)
    else:
        image = np.fromfile(f, dtype='<u1',
                            count=image_size).reshape(-1, row_size)
    # BMPs are saved typically as the last row first.
    # Except if the image height is negative
    if info_header['image_height'] > 0:
        image = image[::-1, :]

    # image format is returned as BGRA, not RGBA
    # this is actually quite costly
    if compression == 'BI_RGB':
        image = image.reshape(image.shape[0], -1, 4)
        image[:, :, [2, 0]] = image[:, :, [0, 2]]
        # Alpha only exists in BITMAPV3INFOHEADER and later
        if info_header['header_size'] <= header_sizes['BITMAPINFOHEADER']:
            image[:, :, 3] = 255
        return image
    elif compression == 'BI_BITFIELDS':
        if bitfields_use_uint8:
            image = image.reshape(image.shape[0], -1, 4)
            if right_shift == [16, 8, 0]:
                image = image[:, :, :3]
                image[:, :, [2, 0]] = image[:, :, [0, 2]]
                return image
        else:
            raw = image.reshape(shape[0], -1)
            if precision == [8, 8, 8]:
                image = np.empty(raw.shape + (3,), dtype=np.uint8)
                for i, r in zip(range(3), right_shift):
                    np.right_shift(raw, r, out=image[:, :, i],
                                   casting='unsafe')
                return image

    raise NotImplementedError(
        "We don't support your particular format yet.")


def _decode_1bpp(f, header, info_header, color_table,
                 shape, row_size):
    f.seek(int(header['file_offset_to_pixelarray']))
    packed_image = np.fromfile(f, dtype='<u1',
                               count=row_size * shape[0]).reshape(-1, row_size)
    if info_header['image_height'] > 0:
        packed_image = packed_image[::-1, :]

    color_index = np.unpackbits(packed_image, axis=1)
    color_index = color_index[:shape[0], :shape[1]]

    color_table = _compress_color_table(color_table)

    return color_table[color_index]


def _decode_24bpp(f, header, info_header, color_table,
                  shape, row_size):
    f.seek(int(header['file_offset_to_pixelarray']))
    image = np.fromfile(f, dtype='<u1',
                        count=row_size * shape[0]).reshape(-1, row_size)
    if info_header['image_height'] > 0:
        image = image[::-1, :]

    image = image.reshape(image.shape[0], -1, 3)
    # image format is returned as BGR, not RGB
    return image[:, :shape[1], ::-1]


def _decode_8bpp(f, header, info_header, color_table,
                 shape, row_size):
    f.seek(int(header['file_offset_to_pixelarray']))
    image = np.fromfile(f, dtype='<u1',
                        count=row_size * shape[0]).reshape(-1, row_size)
    if info_header['image_height'] > 0:
        image = image[::-1, :]

    # These images are padded, make sure you slice them
    image = image[:shape[0], :shape[1]]

    color_table = _compress_color_table(color_table)
    if np.array_equal(color_table, gray_color_table_compressed_uint8):
        return image
    else:
        return color_table[image]


def _decode_4bpp(f, header, info_header, color_table,
                 shape, row_size):
    f.seek(int(header['file_offset_to_pixelarray']))
    packed_image = np.fromfile(f, dtype='<u1',
                               count=row_size * shape[0]).reshape(-1, row_size)
    if info_header['image_height'] > 0:
        packed_image = packed_image[::-1, :]

    color_index = np.empty(shape, dtype=np.uint8)

    # Unpack the image
    out = color_index[:, 0::2]
    np.right_shift(packed_image[:, :out.shape[1]], 4, out=out)
    out = color_index[:, 1::2]
    np.copyto(out, packed_image[:, :out.shape[1]])

    # repeat the color table to index quickly
    table = np.zeros((256 // 2**4, 2**4, color_table.shape[1]), dtype=np.uint8)
    table[:, :color_table.shape[0], :] = color_table
    color_table = table.reshape(-1, color_table.shape[1])

    color_table = _compress_color_table(color_table)

    return color_table[color_index]


def _decode_16bpp(f, header, info_header, color_table,
                 shape, row_size):
    if color_table.size != 0:
        raise NotImplementedError(
            "We don't support colormaps for 16 bit images.")

    compression = compression_types[info_header['compression'][0]]

    if compression == 'BI_BITFIELDS':
        bitfields = np.fromfile(f, dtype='<u4', count=3).tolist()
    else:
        bitfields = BITFIELDS_555

    f.seek(int(header['file_offset_to_pixelarray']))

    image_size = shape[0] * row_size
    image = np.fromfile(f, dtype='<u2',
                        count=image_size // 2).reshape(-1, row_size // 2)
    # BMPs are saved typically as the last row first.
    # Except if the image height is negative
    if info_header['image_height'] > 0:
        image = image[::-1, :]

    packed_image = image[:shape[0], :shape[1]]

    image = np.empty(shape + (3,), dtype=np.uint8)
    if bitfields == BITFIELDS_555:
        np.right_shift(packed_image, 5 + 5, out=image[:, :, 0],
                       casting='unsafe')
        np.right_shift(packed_image, 5, out=image[:, :, 1],
                       casting='unsafe')
        np.copyto(image[:, :, 2], packed_image, casting='unsafe')
        np.take(gray_table_uint5, image, out=image)

    elif bitfields == BITFIELDS_565:
        np.right_shift(packed_image, 5 + 6, out=image[:, :, 0],
                       casting='unsafe')
        np.right_shift(packed_image, 5, out=image[:, :, 1],
                       casting='unsafe')
        np.copyto(image[:, :, 2], packed_image, casting='unsafe')
        np.take(gray_table_uint5, image[:, :, 0::2], out=image[:, :, 0::2])
        np.take(gray_table_uint6, image[:, :, 1], out=image[:, :, 1])
    else:
        raise NotImplementedError(
            "We still haven't implemented your particular bitfield pattern.")

    return image


# Convenient decoder dictionary
_decoders = dict(zip([1, 4, 8, 16, 24, 32],
                     [_decode_1bpp, _decode_4bpp, _decode_8bpp,
                      _decode_16bpp, _decode_24bpp, _decode_32bpp]))


header_t = np.dtype([
    ('signature', '|S2'),
    ('filesize', '<u4'),
    ('reserved1', '<u2'),
    ('reserved2', '<u2'),
    ('file_offset_to_pixelarray', '<u4')
])

bitmap_info_header_t = np.dtype([
    ('header_size', '<u4'),
    ('image_width', '<i4'),
    ('image_height', '<i4'),
    ('image_planes', '<u2'),
    ('bits_per_pixel', '<u2'),
    ('compression', '<u4'),
    ('image_size', '<u4'),
    ('x_pixels_per_meter', '<i4'),
    ('y_pixels_per_meter', '<i4'),
    ('colors_in_color_table', '<u4'),
    ('important_color_count', '<u4'),
])

bitmap_v4_header_t = np.dtype([
    ('header_size', '<u4'),             # 4
    ('image_width', '<i4'),             # 4
    ('image_height', '<i4'),            # 4
    ('image_planes', '<u2'),            # 2
    ('bits_per_pixel', '<u2'),          # 2
    ('compression', '<u4'),             # 4
    ('image_size', '<u4'),              # 4
    ('x_pixels_per_meter', '<i4'),      # 4
    ('y_pixels_per_meter', '<i4'),      # 4
    ('colors_in_color_table', '<u4'),   # 4
    ('important_color_count', '<u4'),   # 4
    ('red_mask', '<u4'),                # 4
    ('green_mask', '<u4'),              # 4
    ('blue_mask', '<u4'),               # 4
    ('alpha_mask', '<u4'),              # 4
    ('color_space', '|S4'),             # 4
    ('cie_xyz_tripple', '<u4', (3, 3)), # 4 * 3 * 3
    ('gamma_red', '<u4'),               # 4
    ('gamma_green', '<u4'),             # 4
    ('gamma_blue', '<u4'),              # 4
])

bitmap_v5_header_t = np.dtype([
    ('header_size', '<u4'),             # 4
    ('image_width', '<i4'),             # 4
    ('image_height', '<i4'),            # 4
    ('image_planes', '<u2'),            # 2
    ('bits_per_pixel', '<u2'),          # 2
    ('compression', '<u4'),             # 4
    ('image_size', '<u4'),              # 4
    ('x_pixels_per_meter', '<i4'),      # 4
    ('y_pixels_per_meter', '<i4'),      # 4
    ('colors_in_color_table', '<u4'),   # 4
    ('important_color_count', '<u4'),   # 4
    ('red_mask', '<u4'),                # 4
    ('green_mask', '<u4'),              # 4
    ('blue_mask', '<u4'),               # 4
    ('alpha_mask', '<u4'),              # 4
    ('color_space', '|S4'),             # 4
    ('cie_xyz_tripple', '<u4', (3, 3)), # 4 * 3 * 3
    ('gamma_red', '<u4'),               # 4
    ('gamma_green', '<u4'),             # 4
    ('gamma_blue', '<u4'),              # 4
    ('intent', '<u4'),                  # 4
    ('profile_data', '<u4'),            # 4
    ('profile_size', '<u4'),            # 4
    ('reserved', '<u4'),                # 4
])

bitmap_core_header_t = np.dtype([
    ('header_size', '<u4'),
    ('image_width', '<u2'),
    ('image_height', '<u2'),
    ('image_planes', '<u2'),
    ('bits_per_pixel', '<u2'),
])

header_sizes = {'BITMAPCOREHEADER':12,
                'BITMAPINFOHEADER': 40,
                'BITMAPV4HEADER': 108,
                'BITMAPV5HEADER': 124}

# bitfields is so poorly documented
# See this post about it
# http://www.virtualdub.org/blog/pivot/entry.php?id=177
BITFIELDS_555 = [0x7C00, 0x03E0, 0x001F]
BITFIELDS_565 = [0xF800, 0x07E0, 0x001F]

info_header_t_dict = {12 : bitmap_core_header_t,
                      40 : bitmap_info_header_t,
                      108: bitmap_v4_header_t,
                      124: bitmap_v5_header_t}

compression_types = ['BI_RGB', 'BI_RLE8', 'BI_RLE4', 'BI_BITFIELDS', 'BI_JPEG',
                     'BI_PNG', 'BI_ALPHABITFIELDS', 'BI_CMYK', 'BI_CMYKRLE8'
                     'BI_CMYKRLE4']

# These are mostly for writing.
gray_color_table_compressed_uint8 = np.arange(256, dtype='<u1')
gray_color_table_uint8 = np.stack([gray_color_table_compressed_uint8,
                                   gray_color_table_compressed_uint8,
                                   gray_color_table_compressed_uint8,
                                   np.full_like(gray_color_table_compressed_uint8,
                                                fill_value=0, dtype='<u1')],
                                  axis=1)

gray_color_table_bool = np.asarray([0, 255], dtype='<u1')
gray_color_table_bool = np.stack([gray_color_table_bool,
                                  gray_color_table_bool,
                                  gray_color_table_bool,
                                  np.full_like(gray_color_table_bool,
                                               fill_value=0, dtype='<u1')],
                                 axis=1)

# Need to convert 16 bit packed numbers to RGB
# These are pretty hacky optimizations
# basically, the size of a 256 bit array is insiginifcant in terms of
# memory consumption. Therefore, we create an array that can be indexed
# while effectively ignoring the most significant bits in the a uint8
gray_table_uint1 = np.asarray([0, 255], dtype='<u1')
gray_table_uint1 = np.concatenate([gray_table_uint1] * (256 // gray_table_uint1.size))

gray_table_uint2 = np.asarray([0, 85, 170, 255], dtype='<u1')
gray_table_uint2 = np.concatenate([gray_table_uint2] * (256 // gray_table_uint2.size))

gray_table_uint3 = np.linspace(0, 255, num=2**3, dtype=np.uint8)
gray_table_uint3 = np.concatenate([gray_table_uint3] * (256 // gray_table_uint3.size))

gray_table_uint4 = np.linspace(0, 255, num=2**4, dtype=np.uint8)
gray_table_uint4 = np.concatenate([gray_table_uint4] * (256 // gray_table_uint4.size))

gray_table_uint5 = np.linspace(0, 255, num=2**5, dtype=np.uint8)
gray_table_uint5 = np.concatenate([gray_table_uint5] * (256 // gray_table_uint5.size))

gray_table_uint6 = np.linspace(0, 255, num=2**6, dtype=np.uint8)
gray_table_uint6 = np.concatenate([gray_table_uint6] * (256 // gray_table_uint6.size))

gray_table_uint7 = np.linspace(0, 255, num=2**7, dtype=np.uint8)
gray_table_uint7 = np.concatenate([gray_table_uint7] * (256 // gray_table_uint7.size))

gray_tables = dict(zip(range(1, 8),
                       [gray_table_uint1,
                        gray_table_uint2,
                        gray_table_uint3,
                        gray_table_uint4,
                        gray_table_uint5,
                        gray_table_uint6,
                        gray_table_uint7]))
