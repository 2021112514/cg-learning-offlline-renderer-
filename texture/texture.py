from PIL import Image
from typing import Dict, Literal
from device import np, Array


# [H, W, 3] 或 [H, W]
images: Dict[str, Array] = {}


def load_image(path: str, mode: Literal['RGB', 'L']):
    images[path] = np.asarray(Image.open(path).convert(mode), dtype=np.uint8)


class Texture:
    def __init__(self, path: str, mode: Literal['RGB', 'L'] = 'RGB'):
        from run import settings
        folder = settings.resources.texture
        if not folder.endswith('/') and not folder.endswith('\\'):
            folder = folder + '/'
        self.path = folder + path
        if self.path not in images.keys():
            load_image(self.path, mode)
        self.image = images[self.path]
        self.mode = mode

    def sample(self, uv: Array) -> Array:
        """
        Args:
            uv(Array): [2, N]，uv坐标
        Returns:
            colors(Array): [3, N] 或 [N]
        """
        if self.mode == 'RGB':
            H, W, _ = self.image.shape
        else:
            H, W = self.image.shape
        # wrap
        uv = uv - np.floor(uv, dtype=np.float32)
        uv[uv < 0] += 1
        x, y = uv * np.array([[W], [H]], dtype=np.float32) - 0.5
        x_floor = np.floor(x, dtype=np.float32)
        y_floor = np.floor(y, dtype=np.float32)
        k1 = (x - x_floor)[:, None] if self.mode == 'RGB' else x - x_floor
        k2 = (y - y_floor)[:, None] if self.mode == 'RGB' else y - y_floor
        x_int = x_floor.astype(np.int32)
        y_int = y_floor.astype(np.int32)
        x_int_1 = (x_int + 1) % W
        y_int_1 = (y_int + 1) % H
        x_int = x_int % W
        y_int = y_int % H
        c1 = self.image[y_int, x_int] * (1 - k1) + self.image[y_int, x_int_1] * k1
        c2 = self.image[y_int_1, x_int] * (1 - k1) + self.image[y_int_1, x_int_1] * k1
        ret = (c1 * (1 - k2) + c2 * k2) / 255
        if self.mode == 'RGB':
            return ret.T
        else:
            return ret