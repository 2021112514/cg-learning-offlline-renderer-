from device import np, Array
from typing import List, Tuple
from .primitive import Primitive
from transform import Transform, identity_transform
from parser import tag


@tag()
class Triangle(Primitive):
    def __init__(
        self,
        vertex: List[Tuple[float, float, float]],
        transform: Transform = identity_transform
    ):
        super().__init__(transform)
        self.p0 = np.array([vertex[0]]).T
        p1_ = np.array([vertex[1]]).T
        p2_ = np.array([vertex[2]]).T
        self.e1 = p1_ - self.p0
        self.e2 = p2_ - self.p0
        self.n = np.cross(self.e1, self.e2, axis=0) # [3, 1]

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array]:
        N = direction.shape[-1]
        s = source - self.p0 # [3, N]
        s1 = np.cross(direction, self.e2, axis=0) # [3, N]
        s2 = np.cross(s, self.e1, axis=0) # [3, N]
        det = self.e1.T @ s1 # [1, N]
        det_neq_0 = np.abs(det) > 1e-8
        k = np.ones((1, N))
        k[:, det_neq_0[0]] = 1 / det[det_neq_0]
        b1 = (k * (s * s1).sum(axis=0))[0] # [N]
        b2 = (k * (s2 * direction).sum(axis=0))[0]
        t = (k * (s2 * self.e2).sum(axis=0))[0]
        in_triangle = (t <= max_t) & (t >= min_t) & det_neq_0[0] & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
        t = np.where(in_triangle, t, np.inf)
        pos = np.full((3, N), np.inf)
        pos[:, in_triangle] = source[:, in_triangle] + t[in_triangle] * direction[:, in_triangle]
        n = self.n.repeat(N, axis=-1)
        n = np.where(in_triangle, n, np.inf)
        return t, pos, n