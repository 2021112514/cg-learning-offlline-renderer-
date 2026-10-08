import math
from typing import Tuple
from device import np, Array
from transform import normalize
from .light import Light
from parser import tag


@tag()
class SpotLight(Light):
    def __init__(
        self,
        direction: Tuple[float, float, float],
        position: Tuple[float, float, float],
        intensity: Tuple[float, float, float],
        inner_angle: float,
        outer_angle: float,
        attenuations: Tuple[float, float, float]
    ):
        self.direction = normalize(np.array([direction]).T)
        self.position = np.array([position]).T
        self.intensity = np.array([intensity]).T
        self.cos_inner_angle = math.cos(inner_angle * np.pi / 180)
        self.cos_outer_angle = math.cos(outer_angle * np.pi / 180)
        self.a = attenuations[0]
        self.b = attenuations[1]
        self.c = attenuations[2]

    def get_radiance(self, pos: Array) -> Tuple[Array, Array]:
        r2 = ((pos - self.position) ** 2).sum(axis=0)
        k1 = 1.0 / np.clip(self.a * r2 + self.b * np.sqrt(r2) + self.c, a_min=1e-8, a_max=None) # [N]
        direc = normalize(pos - self.position)
        cos = self.direction.T @ direc # [1, N]
        k2 = np.clip((cos - self.cos_outer_angle) / (self.cos_inner_angle - self.cos_outer_angle), a_min=0.0, a_max=1.0)
        return self.intensity * k1 * k2, np.broadcast_to(self.position, pos.shape)

    def get_distance(self, pos: Array) -> Array:
        return np.sqrt(((pos - self.position) ** 2).sum(axis=0))