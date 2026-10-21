from typing import Tuple, Optional
from .specular import Specular
from common import get_refractor_vector
from parser import tag
from device import np, Array


@tag()
class DielectricSpecular(Specular):
    def __init__(
        self,
        name: str,
        eta: float,
        transmission_color: Tuple[float, float, float],
        albedo_map: Optional[str] = None,
        normal_map: Optional[str] = None,
        bump_scale: float = 1
    ):
        super().__init__(name, albedo_map, normal_map, bump_scale)
        self.eta = np.float32(eta)
        self.transmission_color = np.array([transmission_color], dtype=np.float32).T

    def _brdf(self, wo: Array, wi: Array) -> Array:
        symm: Array = np.sum(np.abs(wo + wi * np.array([[1.], [1.], [-1.]], dtype=np.float32)) < 1e-6, axis=0) == 3
        cos_i = np.abs(wi[-1, symm])
        eta_i_div_eta_t = np.where(wi[-1, symm] > 0, 1 / self.eta, self.eta)
        fr = np.full_like(cos_i, 1.0)
        wt, refractor = get_refractor_vector(wi[:, symm], eta_i_div_eta_t)
        cos_t = np.abs(wt[-1, :])
        cos_symm = cos_i
        cos_i = cos_i[refractor]
        eta_i_div_eta_t = eta_i_div_eta_t[refractor]
        r1 = (cos_i - eta_i_div_eta_t * cos_t) / (cos_i + eta_i_div_eta_t * cos_t)
        r2 = (eta_i_div_eta_t * cos_i - cos_t) / (eta_i_div_eta_t * cos_i + cos_t)
        fr[refractor] = (r1 ** 2 + r2 ** 2) * 0.5
        ret = np.zeros_like(wi)
        ret[:, symm] = np.repeat(fr[None, :], 3, axis=0) / cos_symm
        return ret

    def btdf(self, wt: Array, wi: Array) -> Array:
        eta_i_div_eta_t = np.where(wi[-1, :] > 0, 1 / self.eta, self.eta)
        wt_, refractor = get_refractor_vector(wi, eta_i_div_eta_t)
        symm = np.full_like(refractor, False)
        symm[refractor] = np.max(np.abs(wt_ - wt[:, refractor]), axis=0) < 1e-6
        cos_i = np.abs(wi[-1, symm])
        cos_t = np.abs(wt[-1, symm])
        eta_i_div_eta_t = eta_i_div_eta_t[symm]
        r1 = (cos_i - eta_i_div_eta_t * cos_t) / (cos_i + eta_i_div_eta_t * cos_t)
        r2 = (eta_i_div_eta_t * cos_i - cos_t) / (eta_i_div_eta_t * cos_i + cos_t)
        fr = (r1 ** 2 + r2 ** 2) * 0.5
        ret = np.zeros_like(wi)
        ret[:, symm] = ((1 - fr) / (eta_i_div_eta_t ** 2 * np.clip(cos_i, a_min=1e-6, a_max=None))) * self.transmission_color
        return ret
