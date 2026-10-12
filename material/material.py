from device import Array


class Material:
    def __init__(self, name: str):
        self.name = name

    def brdf(self, wo: Array, wi: Array) -> Array:
        """
        Bidirectional Reflectance Distribution Function，双向反射分布函数
        Args:
            wo(Array): [3, N] **局部坐标系下**出射方向向量
            wi(Array): [3, N] **局部坐标系下**入射方向向量
        """
        raise NotImplementedError()