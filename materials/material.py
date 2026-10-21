from typing import Tuple, Optional
from device import Array
from texture import Texture


class Material:
    def __init__(self, name: str, texture: Optional[str] = None):
        self.name = name
        self.texture = Texture(texture) if texture is not None else None

    def brdf(self, wo: Array, wi: Array, uv: Array) -> Array:
        """
        Bidirectional Reflectance Distribution Function，双向反射分布函数
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
            wi(Array): [3, N]，**局部坐标系下**入射方向向量
            uv(Array): [2, N]，uv坐标
        """
        brdf = self._brdf(wo, wi)
        if self.texture is not None:
            brdf = brdf * self.texture.sample(uv)
        return brdf

    def _brdf(self, wo: Array, wi: Array) -> Array:
        """
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
            wi(Array): [3, N]，**局部坐标系下**入射方向向量
        """
        raise NotImplementedError()

    def sample_wi(self, wo: Array) -> Tuple[Array, Array]:
        """
        按重要性采样算法采样入射方向，用于计算 Monte Carlo 积分
        Args:
            wo(Array): [3, N]，**局部坐标系下**出射方向向量
        Returns:
            wi(Array): [3, N]，**局部坐标系下**采样的入射方向
            pdf(Array): [N]，采样处的概率密度函数
        """
        raise NotImplementedError()