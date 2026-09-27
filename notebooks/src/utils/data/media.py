import numpy as np
from PIL import Image


def load_image_as_array(image_file, color_format="RGB"):
    image = Image.open(image_file).convert(color_format)
    return np.array(image)
