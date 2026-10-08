from device import np, Array
from typing import Tuple
from .primitive import Primitive
from transform import Transform, identity_transform
from parser import tag


@tag()
class Sphere(Primitive):
    def __init__(
        self,
        radius: float,
        transform: Transform = identity_transform
    ):
        super().__init__(transform)
        self.r = radius

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array]:
        N = direction.shape[-1]
        # A = direction ** 2, direction已归一化
        B = 2 * (source * direction).sum(axis=0) # [N]
        C = (source ** 2).sum(axis=0) - self.r ** 2
        delta = B ** 2 - 4 * C
        real = delta >= 0
        t_large = np.where(real, (-B + np.sqrt(np.clip(delta, a_min=0.0, a_max=None))) / 2, np.inf)
        t_small = np.where(real, (-B - np.sqrt(np.clip(delta, a_min=0.0, a_max=None))) / 2, np.inf)
        t_large = np.where((t_large >= min_t) & (t_large <= max_t), t_large, np.inf)
        t_small = np.where((t_small >= min_t) & (t_small <= max_t), t_small, np.inf)
        t = np.stack([t_large, t_small], axis=0).min(axis=0) # [N]
        not_inf = t < np.inf
        pos = np.full((3, N), np.inf)
        pos[:, not_inf] = source[:, not_inf] + t[not_inf] * direction[:, not_inf]
        # 不能直接返回pos，否则Shape中会对同一个ndarray进行变换
        n = np.full((3, N), np.inf)
        n[:, not_inf] = pos[:, not_inf]
        return t, pos, n