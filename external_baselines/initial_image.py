"""Generation's complete image input interface: exactly one explicit file."""
import numpy as np
from PIL import Image


def read_initial_image(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB').resize((256, 256), Image.Resampling.BICUBIC))
