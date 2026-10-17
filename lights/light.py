from typing import Tuple
from device import Array


class Light:
    def get_radiance(self, pos: Array) -> Tuple[Array, Array, Array, Array]:
        """
        获取pos处的radiance
        Args:
            pos(Array): [3, N]，**世界坐标系下**点的位置
        Returns:
            radiance(Array): [3, N]，radiance
            source(Array): [3, N]，**世界坐标系下**光源坐标
            distance(Array): [N]，`pos`距光源的距离
            pdf(Array): [N]
        """
        raise NotImplementedError()