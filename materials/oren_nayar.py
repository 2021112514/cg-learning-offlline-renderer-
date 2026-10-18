from typing import Tuple
from device import np, Array, random_float32
from .material import Material
from parser import tag

@tag()
class OrenNayar(Material):
    def __init__(self, name: str, albedo: Tuple[float, float, float], sigma: float):
        super().__init__(name)
        self.albedo = np.array([albedo], dtype=np.float32).T
        sigma2 = sigma ** 2
        self.A = 1 - (sigma2 / (2 * sigma2 + 0.33))
        self.B = 0.45 * sigma2 / (sigma2 + 0.09)

    def brdf(self, wo: Array, wi: Array) -> Array:
        cos_theta_i = wi[2, :]
        sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
        cos_phi_i = wi[0, :] / np.clip(sin_theta_i, a_min=1e-6, a_max=None)
        sin_phi_i = wi[1, :] / np.clip(sin_theta_i, a_min=1e-6, a_max=None)
        tan_theta_i = sin_theta_i / np.clip(cos_theta_i, a_min=1e-6, a_max=None)
        cos_theta_o = wo[2, :]
        sin_theta_o = np.sqrt(1 - cos_theta_o ** 2)
        cos_phi_o = wo[0, :] / np.clip(sin_theta_o, a_min=1e-6, a_max=None)
        sin_phi_o = wo[1, :] / np.clip(sin_theta_o, a_min=1e-6, a_max=None)
        tan_theta_o = sin_theta_o / np.clip(cos_theta_o, a_min=1e-6, a_max=None)
        sin_alpha = np.maximum(sin_theta_i, sin_theta_o)
        tan_beta = np.minimum(tan_theta_i, tan_theta_o)
        cos_delta_phi = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o
        return self.albedo * (self.A + self.B * np.clip(cos_delta_phi, a_min=0.0, a_max=None) * sin_alpha * tan_beta) / np.pi

    def sample_wi(self, wo: Array) -> Tuple[Array, Array]:
        N = wo.shape[-1]
        e1 = random_float32(N)
        e2 = random_float32(N)
        cos_theta = np.sqrt(1 - e1)
        sin_theta = np.sqrt(e1)
        phi = 2 * np.pi * e2
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        wi = np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)
        pdf = cos_theta / np.pi
        return wi, pdf