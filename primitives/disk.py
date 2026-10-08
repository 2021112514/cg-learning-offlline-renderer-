from device import np, Array
from typing import Tuple
from .primitive import Primitive
from transform import Transform, identity_transform
from parser import tag


@tag()
class Disk(Primitive):
    """
    xOy平面中的圆盘，圆心在坐标原点，法向量沿z轴正方向
    """
    def __init__(
        self,
        r: float,
        transform: Transform = identity_transform
    ):
        super().__init__(transform)
        self.r = r

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array]:
        N = direction.shape[-1]
        d_z_neq_0 = np.abs(direction[2, :]) > 1e-8
        t = np.full(N, np.inf)
        t[d_z_neq_0] = -source[2, d_z_neq_0] / direction[2, d_z_neq_0]
        t = np.where((t >= min_t) & (t <= max_t), t, np.inf)
        not_inf = t < np.inf
        pos = np.full((3, N), np.inf)
        pos[:, not_inf] = source[:, not_inf] + t[not_inf] * direction[:, not_inf]
        dis2 = np.full(N, np.inf)
        dis2[not_inf] = (pos[:, not_inf] ** 2).sum(axis=0)
        in_disk = dis2 <= self.r ** 2
        t = np.where(in_disk, t, np.inf)
        pos = np.where(in_disk, pos, np.inf)
        n = np.array([[0.], [0.], [1.]]).repeat(N, axis=-1)
        n = np.where(in_disk, n, np.inf)
        return t, pos, n