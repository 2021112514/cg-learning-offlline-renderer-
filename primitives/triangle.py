from device import np, Array
from typing import List, Tuple
from .primitive import Primitive
from common import Transform, identity_transform, sample_triangle
from parser import tag


@tag()
class Triangle(Primitive):
    def __init__(
        self,
        vertex: List[Tuple[float, float, float]] | List[Tuple[float, float, float, float, float]],
        transform: Transform = identity_transform
    ):
        super().__init__(transform)
        self.p0 = np.array([vertex[0][:3]], dtype=np.float32).T
        self.p1 = np.array([vertex[1][:3]], dtype=np.float32).T
        self.p2 = np.array([vertex[2][:3]], dtype=np.float32).T
        self.uv0 = np.array([vertex[0][3:]], dtype=np.float32).T if len(vertex[0]) == 5 else np.zeros((2, 1), dtype=np.float32)
        self.uv1 = np.array([vertex[1][3:]], dtype=np.float32).T if len(vertex[1]) == 5 else np.zeros((2, 1), dtype=np.float32)
        self.uv2 = np.array([vertex[2][3:]], dtype=np.float32).T if len(vertex[2]) == 5 else np.zeros((2, 1), dtype=np.float32)
        self.e1 = self.p1 - self.p0
        self.e2 = self.p2 - self.p0
        self.n = np.cross(self.e1, self.e2, axis=0) # [3, 1]
        self.pdf = 2 / np.sqrt((self.n ** 2).sum())
        du1, dv1 = self.uv1 - self.uv0
        du2, dv2 = self.uv2 - self.uv0
        self.tangent = (dv2 * self.e1 - dv1 * self.e2) / (du1 * dv2 - du2 * dv1)

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array, Array, Array]:
        N = direction.shape[-1]
        s = source - self.p0 # [3, N]
        s1 = np.cross(direction, self.e2, axis=0) # [3, N]
        s2 = np.cross(s, self.e1, axis=0) # [3, N]
        det = self.e1.T @ s1 # [1, N]
        det_neq_0 = np.abs(det) > 1e-8
        k = np.ones((1, N), dtype=np.float32)
        k[:, det_neq_0[0]] = 1 / det[det_neq_0]
        b1 = (k * (s * s1).sum(axis=0))[0] # [N]
        b2 = (k * (s2 * direction).sum(axis=0))[0]
        t = (k * (s2 * self.e2).sum(axis=0))[0]
        in_triangle = (t <= max_t) & (t >= min_t) & det_neq_0[0] & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
        t = np.where(in_triangle, t, np.inf)
        pos = np.full((3, N), np.inf, dtype=np.float32)
        pos[:, in_triangle] = source[:, in_triangle] + t[in_triangle] * direction[:, in_triangle]
        n = np.broadcast_to(self.n, (3, N))
        n = np.where(in_triangle, n, np.inf)
        uv = (1 - b1 - b2) * self.uv0 + b1 * self.uv1 + b2 * self.uv2
        return t, pos, n, uv, np.broadcast_to(self.tangent, (3, N))

    def _intersect_shadow_ray(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Array:
        N = direction.shape[-1]
        s = source - self.p0 # [3, N]
        s1 = np.cross(direction, self.e2, axis=0) # [3, N]
        s2 = np.cross(s, self.e1, axis=0) # [3, N]
        det = self.e1.T @ s1 # [1, N]
        det_neq_0 = np.abs(det) > 1e-8
        k = np.ones((1, N), dtype=np.float32)
        k[:, det_neq_0[0]] = 1 / det[det_neq_0]
        b1 = (k * (s * s1).sum(axis=0))[0] # [N]
        b2 = (k * (s2 * direction).sum(axis=0))[0]
        t = (k * (s2 * self.e2).sum(axis=0))[0]
        in_triangle = (t <= max_t) & (t >= min_t) & det_neq_0[0] & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
        t = np.where(in_triangle, t, np.inf)
        return t

    def _sample(self, n: int) -> Tuple[Array, Array, Array]:
        position = sample_triangle(self.p0, self.p1, self.p2, n)
        normal = np.broadcast_to(self.n, (3, n))
        pdf = np.broadcast_to(self.pdf, n)
        return position, normal, pdf