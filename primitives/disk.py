from device import np, Array
from typing import Tuple
from .primitive import Primitive
from common import Transform, identity_transform, sample_disk
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
        self.pdf = np.array([1 / (np.pi * self.r ** 2)], dtype=np.float32)

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array, Array, Array]:
        N = direction.shape[-1]
        d_z_neq_0 = np.abs(direction[2, :]) > 1e-8
        t = np.full(N, np.inf, dtype=np.float32)
        t[d_z_neq_0] = -source[2, d_z_neq_0] / direction[2, d_z_neq_0]
        t = np.where((t >= min_t) & (t <= max_t), t, np.inf)
        not_inf = t < np.inf
        pos = np.full((3, N), np.inf, dtype=np.float32)
        pos[:, not_inf] = source[:, not_inf] + t[not_inf] * direction[:, not_inf]
        dis2 = np.full(N, np.inf, dtype=np.float32)
        dis2[not_inf] = (pos[:, not_inf] ** 2).sum(axis=0)
        in_disk = dis2 <= self.r ** 2
        t = np.where(in_disk, t, np.inf)
        pos = np.where(in_disk, pos, np.inf)
        n = np.array([[0.], [0.], [1.]], dtype=np.float32).repeat(N, axis=-1)
        n = np.where(in_disk, n, np.inf)
        uv = np.zeros((2, N), dtype=np.float32)
        u = np.arctan2(pos[1, not_inf], pos[0, not_inf]) / (2 * np.pi)
        u[u < 0] += 1
        uv[0, not_inf] = u
        uv[1, not_inf] = np.sqrt(dis2[not_inf]) / self.r
        tangent = np.stack([-pos[1], pos[0], np.zeros(N, np.float32)], axis=0)
        return t, pos, n, uv, tangent

    def _intersect_shadow_ray(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Array:
        N = direction.shape[-1]
        d_z_neq_0 = np.abs(direction[2, :]) > 1e-8
        t = np.full(N, np.inf, dtype=np.float32)
        t[d_z_neq_0] = -source[2, d_z_neq_0] / direction[2, d_z_neq_0]
        t = np.where((t >= min_t) & (t <= max_t), t, np.inf)
        not_inf = t < np.inf
        pos = np.full((3, N), np.inf, dtype=np.float32)
        pos[:, not_inf] = source[:, not_inf] + t[not_inf] * direction[:, not_inf]
        dis2 = np.full(N, np.inf, dtype=np.float32)
        dis2[not_inf] = (pos[:, not_inf] ** 2).sum(axis=0)
        in_disk = dis2 <= self.r ** 2
        t = np.where(in_disk, t, np.inf)
        return t

    def _sample(self, n: int) -> Tuple[Array, Array, Array]:
        position = np.concat([sample_disk(self.r, n), np.zeros((1, n), dtype=np.float32)], axis=0)
        normal = np.broadcast_to(np.array([[0.0], [0.0], [1.0]], dtype=np.float32), (3, n))
        pdf = np.broadcast_to(self.pdf, n)
        return position, normal, pdf