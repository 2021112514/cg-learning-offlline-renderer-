from typing import Tuple
from device import np, Array
from .light import Light
from scene.scene_object import SceneObject


class AreaLight(Light):
    def __init__(self, scene_object: SceneObject):
        self.scene_object = scene_object

    def get_radiance(self, pos: Array) -> Tuple[Array, Array, Array, Array]:
        N = pos.shape[-1]
        source, normal, pdf = self.scene_object.sample(N)
        d = pos - source
        distance = np.sqrt(np.sum(d ** 2, axis=0))
        d /= distance
        cos_thata_a = np.sum(d * normal, axis=0)
        cos_leq0 = cos_thata_a < 1e-6
        radiance = np.where(cos_leq0[None, :], np.array([[0.0], [0.0], [0.0]]), np.array([self.scene_object.emissive]).T)
        pdf_transformed = distance ** 2 * pdf / np.clip(cos_thata_a, a_min=1e-6, a_max=1.0)
        return radiance, source, distance - 1e-6, pdf_transformed