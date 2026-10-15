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

    def sample_wi(self, wo: Array) -> Tuple[Array, Array]:
        N = wo.shape[-1]
        e1 = np.random.random(N)
        e2 = np.random.random(N)
        cos_theta = np.sqrt(1 - e1)
        sin_theta = np.sqrt(e1)
        phi = 2 * np.pi * e2
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        wi = np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)
        pdf = cos_theta / np.pi
        return wi, pdf