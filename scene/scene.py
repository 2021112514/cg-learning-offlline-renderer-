import xml.sax
from device import np, Array
from typing import List, Tuple
from .scene_object import SceneObject
from material import *
from ray import Ray, PerspectiveCamera
from light import Light
from parser import tag, XMLHandler


@tag()
class Scene:
    def __init__(
        self,
        scene_objects: List[SceneObject],
        lights: List[Light],
        camera: PerspectiveCamera,
        materials: List[Material]
    ):
        self.objects = scene_objects
        self.lights = lights
        self.camera = camera
        self.materials = materials
        material_map = {
            m.name: i for i, m in enumerate(self.materials)
        }
        self.object_material_index_map = np.array([material_map[obj.material] for obj in self.objects])

    def intersect(self, ray: Ray) -> Tuple[Array, Array, Array, Array]:
        """
        Returns:
            t(Array): [N]，交点距射线原点的距离
            pos(Array): [3, N]，交点坐标
            n(Array): [3, N]，交点法线
            object_index(Array): [N]，与射线最先相交的`SceneObject`的索引，用于获取交点材质
        """
        N = ray.direction.shape[-1]
        arange = np.arange(N)
        old_t = np.full(N, np.inf)
        old_pos = np.full((3, N), np.inf)
        old_n = np.full((3, N), np.inf)
        object_index = np.full(N, -1, dtype=np.int32)
        for i, object in enumerate(self.objects):
            t, pos, n = object.intersect(ray)
            t_stack = np.stack([old_t, t], axis=0)
            pos_stack = np.stack([old_pos, pos], axis=0)
            n_stack = np.stack([old_n, n], axis=0)
            index = np.argmin(t_stack, axis=0)
            object_index[index == 1] = i
            old_t = t_stack[index, arange]
            old_pos = pos_stack[index, :, arange].T
            old_n = n_stack[index, :, arange].T
        return old_t, old_pos, old_n, object_index

    @classmethod
    def from_xml(cls, path: str) -> 'Scene':
        parser = xml.sax.make_parser()
        handler = XMLHandler()
        parser.setContentHandler(handler)
        parser.parse(path)
        return handler.root