from typing import Tuple
from device import np, Array
from .light import Light
from parser import tag


@tag()
class PointLight(Light):
    def __init__(
        self,
        position: Tuple[float, float, float],
        intensity: Tuple[float, float, float],
        attenuations: Tuple[float, float, float]
    ):
        self.position = np.array([position]).T
        self.intensity = np.array([intensity]).T
        self.a = attenuations[0]
        self.b = attenuations[1]
        self.c = attenuations[2]

    def get_radiance(self, pos: Array) -> Tuple[Array, Array]:
        r2 = ((pos - self.position) ** 2).sum(axis=0)
        radiance = self.intensity / np.clip(self.a * r2 + self.b * np.sqrt(r2) + self.c, a_min=1e-8, a_max=None)
        return radiance, np.broadcast_to(self.position, pos.shape)

    def get_distance(self, pos: Array) -> Array:
        return np.sqrt(((pos - self.position) ** 2).sum(axis=0))