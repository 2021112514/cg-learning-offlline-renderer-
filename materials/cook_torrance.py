from typing import Tuple, Optional
from .material import Material
from common import normalize
from texture import Texture
from device import np, Array, random_float32, EPSILON


class CookTorrance(Material):
    def __init__(
        self,
        name: str,
        albedo: Tuple[float, float, float],
        roughness: float,
        metallic: float,
        albedo_map: Optional[str] = None,
        normal_map: Optional[str] = None,
        bump_scale: float = 1,
        depth_map: Optional[str] = None,
        max_depth: float = 1.0,
        roughness_map: Optional[str] = None,
        metallic_map: Optional[str] = None
    ):
        super().__init__(name, albedo_map, normal_map, bump_scale, depth_map, max_depth)
        self.albedo = np.array([albedo], dtype=np.float32).T
        self.roughness = roughness
        self.metallic = metallic
        self.roughness_map = Texture(roughness_map, 'L') if roughness_map is not None else None
        self.metallic_map = Texture(metallic_map, 'L') if metallic_map is not None else None

    def _brdf(self, wo: Array, wi: Array, uv: Array) -> Array:
        valid = np.logical_and(wi[-1, :] > 0, wo[-1, :] > 0)
        ret = np.zeros_like(wo, dtype=np.float32)
        wh = normalize(wo[:, valid] + wi[:, valid])
        albedo = self.albedo * self.albedo_map.sample(uv[:, valid]) if self.albedo_map is not None else self.albedo
        roughness = self.roughness * self.roughness_map.sample(uv[:, valid]) if self.roughness_map is not None else self.roughness
        metallic = self.metallic * self.metallic_map.sample(uv[:, valid]) if self.metallic_map is not None else self.metallic
        alpha2 = roughness ** 4
        D = alpha2 / np.clip(np.pi * ((alpha2 - 1) * wh[-1, :] ** 2 + 1) ** 2, a_min=EPSILON, a_max=None)
        F0 = 0.04 * (1 - metallic) + albedo * metallic
        cos_h_v = np.clip(np.sum(wo[:, valid] * wh, axis=0), a_min=0.0, a_max=1.0)
        F = F0 + (1 - F0) * np.pow(1 - cos_h_v, 5)
        k = (roughness + 1) ** 2 / 8
        G = self._geometry_schlick_ggx(k, np.abs(wi[-1, valid])) * self._geometry_schlick_ggx(k, np.abs(wo[-1, valid]))
        kd = (1 - F) * (1 - metallic)
        ret[:, valid] = kd * albedo / np.pi + F * G * D / (4 * wo[-1, valid] * wi[-1, valid])
        return ret

    def sample_wi(self, wo: Array, uv: Array) -> Tuple[Array, Array]:
        roughness = self.roughness * self.roughness_map.sample(uv) if self.roughness_map is not None else self.roughness
        weight = 0.5
        N = wo.shape[1]
        pdf = np.zeros(N, dtype=np.float32)
        wi = np.zeros((3, N), dtype=np.float32)
        less_than_weight = random_float32(N) < weight
        arange = np.arange(N)
        index_ggx = arange[less_than_weight]
        index_lambert = arange[np.logical_not(less_than_weight)]
        n_sample_ggx = index_ggx.shape[0]
        e1 = random_float32(N)
        e2 = random_float32(N)
        # 按GGX项重要性采样的部分
        if n_sample_ggx > 0:
            e11 = e1[:n_sample_ggx]
            e21 = e2[:n_sample_ggx]
            roughness_gxx = roughness if isinstance(roughness, float | int) else roughness[index_ggx]
            alpha2 = roughness_gxx ** 4
            cos_theta_h2 = np.clip((1 - e11) / np.clip(e11 * (alpha2 - 1) + 1, a_min=EPSILON, a_max=None), a_min=0, a_max=None)
            cos_theta_h = np.sqrt(cos_theta_h2)
            sin_theta_h = np.sqrt(np.clip(1 - cos_theta_h2, a_min=0, a_max=None))
            phi_h = e21 * 2 * np.pi
            wh_ggx = np.stack([np.cos(phi_h) * sin_theta_h, np.sin(phi_h) * sin_theta_h, cos_theta_h], axis=0)
            wo_ggx = wo[:, index_ggx]
            wo_dot_wh = np.sum(wo_ggx * wh_ggx, axis=0)
            wi_ggx = 2 * wo_dot_wh * wh_ggx - wo_ggx
            wi[:, index_ggx] = wi_ggx
            pdf[index_ggx] += weight * self._distribution_ggx(roughness_gxx, wh_ggx) * cos_theta_h / np.clip(4 * wo_dot_wh, a_min=EPSILON, a_max=None)
            pdf[index_ggx] += (1 - weight) * wi_ggx[-1, :] / np.pi
            pdf[index_ggx] *= np.logical_and(wo_dot_wh > 0, wi_ggx[-1, :] > 0)
        # 按lambert项重要性采样的部分
        if n_sample_ggx < N:
            e12 = e1[n_sample_ggx:]
            e22 = e2[n_sample_ggx:]
            cos_theta = np.sqrt(1 - e12)
            sin_theta = np.sqrt(e12)
            phi = e22 * 2 * np.pi
            wi_lambert = np.stack([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta], axis=0)
            wi[:, index_lambert] = wi_lambert
            wo_lambert = wo[:, index_lambert]
            wh_lambert = normalize(wo_lambert + wi_lambert)
            wo_dot_wh = np.sum(wo_lambert * wh_lambert, axis=0)
            roughness_lambert = roughness if isinstance(roughness, float | int) else roughness[index_lambert]
            pdf[index_lambert] += weight * self._distribution_ggx(roughness_lambert, wh_lambert) * wh_lambert[-1, :] \
                / np.clip(4 * wo_dot_wh, a_min=EPSILON, a_max=None)
            pdf[index_lambert] *= np.logical_and(wo_dot_wh > 0, wh_lambert[-1, :] > 0)
            pdf[index_lambert] += (1 - weight) * cos_theta / np.pi
        return wi, pdf

    def _geometry_schlick_ggx(self, k: float | Array, n_dot_v: Array):
        return n_dot_v / np.clip(n_dot_v * (1 - k) + k, a_min=EPSILON, a_max=None)

    def _distribution_ggx(self, roughness: float | Array, wh: Array):
        a = roughness ** 2
        a2 = a ** 2
        return a2 / np.clip(np.pi * (wh[-1, :] ** 2 * (a2 - 1) + 1) ** 2, a_min=EPSILON, a_max=None)

    def process_albedo_map(self, uv: Array, brdf: Array) -> Array:
        return brdf