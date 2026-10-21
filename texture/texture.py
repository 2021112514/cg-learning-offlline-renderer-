from PIL import Image
from typing import Dict
from device import np, Array


# [H, W, 3]
images: Dict[str, Array] = {}


def load_image(path: str):
    images[path] = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


class Texture:
    def __init__(self, path: str):
        from run import settings
        folder = settings.resources.texture
        if not folder.endswith('/') and not folder.endswith('\\'):
            folder = folder + '/'
        self.path = folder + path
        if self.path not in images.keys():
            load_image(self.path)
        self.image = images[self.path]

    def sample(self, uv: Array) -> Array:
        """
        Args:
            uv(Array): [2, N]，uv坐标
        Returns:
            colors(Array): [3, N]
        """
        H, W, _ = self.image.shape
        # wrap
        uv = uv - np.array(uv, dtype=np.int32)
        uv[uv < 0] += 1
        x, y = uv * np.array([[W], [H]], dtype=np.float32) - 0.5
        x_int = np.floor(x, dtype=np.int32)
        y_int = np.floor(y, dtype=np.int32)
        k1 = (x - x_int)[:, None]
        k2 = (y - y_int)[:, None]
        x_int_1 = x_int + 1 % W
        y_int_1 = y_int + 1 % H
        x_int = x_int % W
        y_int = y_int % H
        c1 = self.image[y_int, x_int] * (1 - k1) + self.image[y_int, x_int_1] * k1
        c2 = self.image[y_int_1, x_int] * (1 - k1) + self.image[y_int_1, x_int_1] * k1
        ret = (c1 * (1 - k2) + c2 * k2) / 255
        return ret.T