from typing import Tuple
from device import Array


class Light:
    def get_radiance(self, pos: Array) -> Tuple[Array, Array]:
        """
        获取pos处的radiance
        Args:
            pos(Array): [3, N]，**世界坐标系下**点的位置
        Returns:
            radiance(Array): [3, N]，radiance
            source(Array): [3, N]，**世界坐标系下**光源坐标
        """
        raise NotImplementedError()

    def get_distance(self, pos: Array) -> Array:
        """
        计算pos距光源的距离
        Args:
            pos(Array): [3, N]，**世界坐标系下**点的位置
        Returns:
            distance(Array): [N]，`pos`距光源的距离
        """
        raise NotImplementedError()