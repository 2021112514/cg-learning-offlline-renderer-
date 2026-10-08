from device import np, Array
from typing import List, Tuple
from primitives import Primitive
from transform import Transform, identity_transform, world_transform
from ray import Ray
from parser import tag


@tag()
class SceneObject:
    def __init__(
        self,
        primitives: List[Primitive],
        material: str,
        transform: Transform = identity_transform
    ):
        o2w, w2o = world_transform(transform.position, transform.rotation, transform.scale)
        for shape in primitives:
            if shape.scene_object is not None:
                raise RuntimeError('Shape must not be reused between scene objects')
            shape.scene_object = self
            shape.o2w = o2w @ shape.o2w
            shape.w2o = shape.w2o @ w2o
        self.shapes = primitives
        self.material = material

    def intersect(self, ray: Ray) -> Tuple[Array, Array, Array]:
        N = ray.direction.shape[-1]
        arange = np.arange(N)
        old_t = np.full(N, np.inf)
        old_pos = np.full((3, N), np.inf)
        old_n = np.full((3, N), np.inf)
        for shape in self.shapes:
            t, pos, n = shape.intersect(ray)
            t_stack = np.stack([old_t, t], axis=0)
            pos_stack = np.stack([old_pos, pos], axis=0)
            n_stack = np.stack([old_n, n], axis=0)
            index = np.argmin(t_stack, axis=0)
            old_t = t_stack[index, arange]
            old_pos = pos_stack[index, :, arange].T
            old_n = n_stack[index, :, arange].T
        return old_t, old_pos, old_n