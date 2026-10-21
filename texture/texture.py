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
        x, y = uv * np.array([[W], [H]], dtype=np.float32)
        x_int = np.floor(x - 0.5, dtype=np.uint32)
        y_int = np.floor(y - 0.5, dtype=np.uint32)
        k1 = (x - x_int - 0.5)[:, None]
        k2 = (y - y_int - 0.5)[:, None]
        x_int_1 = np.clip(x_int + 1, a_min=None, a_max=W - 1)
        y_int_1 = np.clip(y_int + 1, a_min=None, a_max=H - 1)
        x_int = np.clip(x_int, a_min=0, a_max=None)
        y_int = np.clip(y_int, a_min=0, a_max=None)
        c1 = self.image[y_int, x_int] * k1 + self.image[y_int, x_int_1] * (1 - k1)
        c2 = self.image[y_int_1, x_int] * k1 + self.image[y_int_1, x_int_1] * (1 - k1)
        ret = (c1 * k2 + c2 * (1 - k2)) / 255
        return ret.T