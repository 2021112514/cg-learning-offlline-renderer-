from device import Array


class Material:
    def __init__(self, name: str):
        self.name = name

    def brdf(self, wo: Array, wi: Array) -> Array:
        """
        Bidirectional Reflectance Distribution Function，双向反射分布函数
        """
        raise NotImplementedError()