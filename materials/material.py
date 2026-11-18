from typing import Tuple, Optional
from device import np, Array
from texture import Texture


class Material:
    def __init__(
        self,
        name: str,
        texture: Optional[str] = None,
        normal_map: Optional[str] = None,
        bump_scale: float = 1,
        depth_map: Optional[str] = None,
        max_depth: float = 1.0
    ):
        self.name = name
        self.albedo_map = Texture(texture) if texture is not None else None
        self.normal_map = Texture(normal_map) if normal_map is not None else None
        self.depth_map = Texture(depth_map, mode='L') if depth_map is not None else None
        self.bump_scale = bump_scale
        self.max_depth = max_depth
        from run import settings
        self.depth_layers = settings.render.depth_layers

    def brdf(self, wo: Array, wi: Array, uv: Array) -> Array:
        """
        Bidirectional Reflectance Distribution Function，双向反射分布函数
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
            wi(Array): [3, N]，**局部坐标系下**入射方向向量
            uv(Array): [2, N]，uv坐标
        """
        # 指定了Primitive的uv时，wo是TBN坐标系下的出射方向向量
        # 视差遮蔽映射(POM)
        if self.depth_map is not None:
            q_a = uv.copy()
            f_a = self.depth_map.sample(uv)
            q_b = np.zeros_like(uv, dtype=np.float32)
            f_b = np.ones_like(f_a, dtype=np.float32)
            for i in range(1, self.depth_layers + 1):
                d = i / self.depth_layers
                duv = (wo[-1, :] > 0) * d * self.max_depth * wo[:2, :] / np.clip(wo[-1, :], a_min=1e-2, a_max=None)
                uv_ = uv - duv
                D = self.depth_map.sample(uv_)
                f = D - d
                not_set = f_b > 0
                mask_a = np.logical_and(f >= 0, not_set)
                f_a[mask_a] = f[mask_a]
                q_a[:, mask_a] = uv_[:, mask_a]
                mask_b = np.logical_and(f <= 0, not_set)
                f_b[mask_b] = f[mask_b]
                q_b[:, mask_b] = uv_[:, mask_b]
            t = -f_a / np.clip(f_b - f_a, a_min=None, a_max=-1e-4)
            uv = (1 - t) * q_a + t * q_b
        brdf = self._brdf(wo, wi, uv)
        return self.process_albedo_map(uv, brdf)

    def process_albedo_map(self, uv: Array, brdf: Array) -> Array:
        if self.albedo_map is not None:
            return brdf * self.albedo_map.sample(uv)
        return brdf

    def _brdf(self, wo: Array, wi: Array, uv: Array) -> Array:
        """
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
            wi(Array): [3, N]，**局部坐标系下**入射方向向量
        """
        raise NotImplementedError()

    def sample_wi(self, wo: Array, uv: Array) -> Tuple[Array, Array]:
        """
        按重要性采样算法采样入射方向，用于计算 Monte Carlo 积分
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
        Returns:
            wi(Array): [3, N]，**局部坐标系下**采样的入射方向
            pdf(Array): [N]，采样处的概率密度函数，**特别地，pdf=0表示采样失败，求解 Monte Carlo 积分时应将对应的共享记为0**
        """
        raise NotImplementedError()