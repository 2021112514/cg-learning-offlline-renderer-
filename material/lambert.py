from device import np, Array
from typing import Tuple
from .material import Material
from parser import tag


@tag()
class Lambert(Material):
    def __init__(self, albedo: Tuple[float, float, float], name: str):
        super().__init__(name)
        self.albedo = np.array([albedo]).T

    def brdf(self, wo: Array, wi: Array) -> Array:
        return np.broadcast_to(self.albedo / np.pi, wo.shape)