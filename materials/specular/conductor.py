from typing import Tuple, Optional
from .specular import Specular
from parser import tag
from device import np, Array


@tag()
class ConductorSpecular(Specular):
    def __init__(
        self,
        name: str,
        eta: Tuple[float, float, float],
        absorption_coef: Tuple[float, float, float],
        reflection_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        albedo_map: Optional[str] = None,
        normal_map: Optional[str] = None,
        bump_scale: float = 1
    ):
        super().__init__(name, albedo_map, normal_map, bump_scale)
        self.eta = np.array([eta], dtype=np.float32).T
        self.reflection_color = np.array([reflection_color], dtype=np.float32).T
        self.k = self.eta ** 2 + np.array([absorption_coef], dtype=np.float32).T ** 2

    def _brdf(self, wo: Array, wi: Array) -> Array:
        symm = np.sum(np.abs(wo + wi * np.array([[1.], [1.], [-1.]], dtype=np.float32)) < 1e-6, axis=0) == 3
        cos = wi[2, symm]
        cos2 = cos ** 2
        a = self.k * cos2 + 1
        b = self.k + cos2
        eta_cos_2 = 2 * self.eta * cos
        r1 = (a - eta_cos_2) / (a + eta_cos_2)
        r2 = (b - eta_cos_2) / (b + eta_cos_2)
        fr = (r1 + r2) * np.float32(0.5)
        ret = np.zeros_like(wi)
        ret[:, symm] = self.reflection_color * fr / np.clip(cos, a_min=1e-6, a_max=None)
        return ret