from device import np, Array
from typing import List, Tuple
from primitives import Primitive
from common import Transform, identity_transform, world_transform
from ray import Ray
from parser import tag


@tag()
class SceneObject:
    def __init__(
        self,
        primitives: List[Primitive],
        material: str,
        emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform: Transform = identity_transform
    ):
        o2w, w2o = world_transform(transform.position, transform.rotation, transform.scale)
        for primitive in primitives:
            if primitive.scene_object is not None:
                raise RuntimeError('Primitive must not be reused between scene objects')
            primitive.scene_object = self
            primitive.o2w = o2w @ primitive.o2w
            primitive.w2o = primitive.w2o @ w2o
            primitive.det_w *= transform.scale[0] * transform.scale[1] * transform.scale[2] if \
                isinstance(transform.scale, tuple) else transform.scale ** 3
        self.primitives = primitives
        self.emissive = emissive
        self.material = material

    def intersect(self, ray: Ray) -> Tuple[Array, Array, Array, Array, Array]:
        N = ray.direction.shape[-1]
        old_t = np.full(N, np.inf, dtype=np.float32)
        old_pos = np.full((3, N), np.inf, dtype=np.float32)
        old_n = np.full((3, N), np.inf, dtype=np.float32)
        old_uv = np.zeros((2, N), dtype=np.float32)
        old_tangent = np.zeros((3, N), dtype=np.float32)
        for primitive in self.primitives:
            t, pos, n, uv, tangent = primitive.intersect(ray)
            closer = t < old_t
            old_t = np.where(closer, t, old_t)
            old_pos = np.where(closer, pos, old_pos)
            old_n = np.where(closer, n, old_n)
            old_uv = np.where(closer, uv, old_uv)
            old_tangent = np.where(closer, tangent, old_tangent)
        return old_t, old_pos, old_n, old_uv, old_tangent

    def intersect_shadow_ray(self, ray: Ray) -> Array:
        N = ray.direction.shape[-1]
        old_t = np.full(N, np.inf, dtype=np.float32)
        for primitive in self.primitives:
            t = primitive.intersect_shadow_ray(ray)
            old_t = np.minimum(t, old_t)
        return old_t

    def sample(self, n: int) -> Tuple[Array, Array, Array]:
        # 遵循教程：为简单起见在primitive间均匀采样，不考虑重要性采样
        index = np.random.randint(0, len(self.primitives), (n))
        position = np.zeros((3, n), dtype=np.float32)
        normal = np.zeros((3, n), dtype=np.float32)
        pdf = np.ones(n, dtype=np.float32)
        for i in range(len(self.primitives)):
            mask = index == i
            num = int(np.sum(mask))
            position[:, mask], normal[:, mask], pdf[mask] = self.primitives[i].sample(num)
        pdf /= len(self.primitives)
        return position, normal, pdf