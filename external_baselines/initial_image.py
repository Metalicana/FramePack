"""Generation's complete image input interface: exactly one explicit file."""
import numpy as np
from PIL import Image


def spatial_layout(width, height, mode='stretch'):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    if mode == 'stretch':
        return (256, 256), (0, 0)
    if mode != 'letterbox':
        raise ValueError(f'Unsupported spatial mode: {mode}')
    scale = min(256 / width, 256 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return size, ((256 - size[0]) // 2, (256 - size[1]) // 2)


def read_initial_image(path, mode='stretch'):
    with Image.open(path) as image:
        size, offset = spatial_layout(*image.size, mode)
        resized = image.convert('RGB').resize(size, Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (256, 256))
        canvas.paste(resized, offset)
        return np.array(canvas)
