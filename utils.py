from PIL import Image
import io
import numpy as np


def resize_image(image: Image, length: int) -> Image:
    """
    Resize an image to a square. Can make an image bigger to make it fit or smaller if it doesn't fit. It also crops
    part of the image.

    :param self:
    :param image: Image to resize.
    :param length: Width and height of the output image.
    :return: Return the resized image.
    """

    """
    Resizing strategy : 
     1) We resize the smallest side to the desired dimension (e.g. 1080)
     2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
    """
    if image.size[0] < image.size[1]:
        # The image is in portrait mode. Height is bigger than width.

        # This makes the width fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((length, int(image.size[1] * (length / image.size[0]))))

        # Amount of pixel to lose in total on the height of the image.
        required_loss = (resized_image.size[1] - length)

        # Crop the height of the image so as to keep the center part.
        resized_image = resized_image.crop(
            box=(0, required_loss / 2, length, resized_image.size[1] - required_loss / 2))

        # We now have a length*length pixels image.
        return resized_image
    else:
        # This image is in landscape mode or already squared. The width is bigger than the heihgt.

        # This makes the height fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((int(image.size[0] * (length / image.size[1])), length))

        # Amount of pixel to lose in total on the width of the image.
        required_loss = resized_image.size[0] - length

        # Crop the width of the image so as to keep 1080 pixels of the center part.
        resized_image = resized_image.crop(
            box=(required_loss / 2, 0, resized_image.size[0] - required_loss / 2, length))

        # We now have a length*length pixels image.
        return resized_image


def load_img(
    path,
    grayscale=False,
    color_mode="rgb",
    target_size=None,
    interpolation="nearest",
    keep_aspect_ratio=False,
):
    if grayscale:
        warnings.warn(
            'grayscale is deprecated. Please use color_mode = "grayscale"'
        )
        color_mode = "grayscale"
    if Image is None:
        raise ImportError(
            "Could not import PIL.Image. The use of `load_img` requires PIL."
        )
    if isinstance(path, io.BytesIO):
        img = Image.open(path)
    elif isinstance(path, (pathlib.Path, bytes, str)):
        if isinstance(path, pathlib.Path):
            path = str(path.resolve())
        with open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
    else:
        raise TypeError(
            f"path should be path-like or io.BytesIO, not {type(path)}"
        )

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            resample = Image.Resampling.NEAREST

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img

def img_to_array(img, data_format='channels_last', dtype=None):

    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x