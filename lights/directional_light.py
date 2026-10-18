from typing import Tuple
from device import np, Array
from common import normalize
from .light import Light
from parser import tag


@tag()
class DirectionalLight(Light):
    def __init__(
        self,
        direction: Tuple[float, float, float],
        radiance: Tuple[float, float, float]
    ):
        self.direction = normalize(np.array([direction], dtype=np.float32).T)
        self.intensity = np.array([radiance], dtype=np.float32).T

    def get_radiance(self, pos: Array) -> Tuple[Array, Array, Array, Array]:
        N = pos.shape[-1]
        return (
            np.broadcast_to(self.intensity, pos.shape),
            pos - 1e6 * self.direction,
            np.broadcast_to(np.array([np.inf], dtype=np.float32), N),
            np.broadcast_to(np.ones(1, dtype=np.float32), N)
        )