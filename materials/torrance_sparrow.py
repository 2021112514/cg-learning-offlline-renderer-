from typing import Tuple, Optional
from device import np, Array, random_float32, EPSILON
from common import normalize
from .material import Material
from parser import tag


@tag()
class TorranceSparrow(Material):
    def __init__(
        self,
        name: str,
        reflection_color: Tuple[float, float, float],
        eta: Tuple[float, float, float],
        absorption_coef: Tuple[float, float, float],
        exponent: float,
        albedo_map: Optional[str]
    ):
        super().__init__(name, albedo_map)
        self.eta = np.array([eta], dtype=np.float32).T
        self.reflection_color = np.array([reflection_color], dtype=np.float32).T
        self.k = self.eta ** 2 + np.array([absorption_coef], dtype=np.float32).T ** 2
        self.e = exponent

    def _brdf(self, wo: Array, wi: Array) -> Array:
        valid = np.logical_and(wo[2, :] > EPSILON, wi[2, :] > EPSILON)
        valid = np.full(wo.shape[-1], True, dtype=bool)
        brdf = np.zeros_like(wo)
        wi_valid = wi[:, valid]
        wo_valid = wo[:, valid]
        wh = normalize(wi_valid + wo_valid)
        D = (self.e + 2) * np.clip(wh[2, :], a_min=np.float32(0.0), a_max=None) ** self.e / (2 * np.pi)
        deno = np.maximum(EPSILON, np.sum(wo_valid * wh, axis=0))
        G = np.minimum(np.ones(1, dtype=np.float32), 2 * wh[2, :] * np.minimum(wo_valid[2, :], wi_valid[2, :]) / deno)
        cos = np.sum(wi_valid* wh, axis=0)
        cos2 = cos ** 2
        a = self.k * cos2 + 1
        b = self.k + cos2
        eta_cos_2 = 2 * self.eta * cos
        r1 = (a - eta_cos_2) / (a + eta_cos_2)
        r2 = (b - eta_cos_2) / (b + eta_cos_2)
        F = (r1 + r2) * np.float32(0.5)
        brdf[:, valid] = self.reflection_color * D * G * F / (4 * wo_valid[2, :] * wi_valid[2, :])
        return brdf

    def sample_wi(self, wo: Array) -> Tuple[Array, Array]:
        N = wo.shape[-1]
        e1 = random_float32(N)
        e2 = random_float32(N)
        cos_theta_h = e1 ** (1 / (self.e + 1))
        sin_theta_h = np.sqrt(1 - cos_theta_h ** 2)
        phi_h = 2 * np.pi * e2
        wh = np.stack([sin_theta_h * np.cos(phi_h), sin_theta_h * np.sin(phi_h), cos_theta_h], axis=0)
        wi = 2 * np.sum(wo * wh, axis=0) * wh - wo
        pdf_wh = (self.e + 1) * (cos_theta_h ** self.e) / (2 * np.pi)
        pdf = pdf_wh / np.clip(4 * np.sum(wo * wh, axis=0), a_min=EPSILON, a_max=None)
        return wi, pdf