from device import np, Array
from typing import Tuple, Any
from ray import Ray
from transform import Transform, world_transform, normalize, expand0, expand1
from parser import tag


@tag()
class Primitive:
    def __init__(
        self,
        transform: Transform
    ):
        self.o2w, self.w2o = world_transform(transform.position, transform.rotation, transform.scale)
        self.scene_object: Any = None

    def intersect(self, ray: Ray) -> Tuple[Array, Array, Array]:
        """
        Returns:
            t(Array): [N]，交点与摄像机的距离，若无交点则为np.inf
            pos(Array): [3, N]，交点位置，若无交点则为[np.inf, np.inf, np.inf]
            n(Array): [3, N]，交点法线
        """
        direction_o_unorm = (self.w2o @ expand0(ray.direction))[:3, :]
        # 摄像机每条光线方向向量变换后的长度，[N]
        norm = np.sqrt((direction_o_unorm ** 2).sum(axis=0))
        direction_o = direction_o_unorm / norm

        # [3, N]
        source = np.broadcast_to(ray.pos, ray.direction.shape)
        source_pos_o = (self.w2o @ expand1(source))[:3, :]

        # 世界坐标下相机到近裁剪平面每个点的距离 变换为 物体坐标下相机到近裁剪平面每个点的距离
        # 注：共线的向量在坐标变换前后的长度之比不变，因为坐标变换是线性变换
        min_t = ray.min_t * norm
        max_t = ray.max_t * norm
        t_o, pos_o, n_o = self._intersect(direction_o, source_pos_o, min_t, max_t)
        not_inf = t_o < np.inf
        # 将物体坐标系下相机到交点的距离变换为世界坐标系下相机到交点的距离
        t_o[not_inf] = t_o[not_inf] / norm[not_inf]
        pos_o[:, not_inf] = (self.o2w @ expand1(pos_o[:, not_inf]))[:3, :]

        # 注意：法线不能直接使用o2w变换，因为缩放矩阵的对角元不同时会使原本正交的向量在变换后变得不正交
        # 物体坐标系中的法向量n'与u正交：(n')^Tu=0，世界坐标系中的n应满足n^T o2w u=0
        # 故n^T o2w = k (n')^T，不妨令k=1，则n=((o2w)^{-1})^T n'=(w2o)^T n'
        # 实际上，若缩放矩阵对角元不同，则两坐标系下同一点处的法向量不同
        n_o[:, not_inf] = normalize((self.w2o.T @ expand0(n_o[:, not_inf]))[:3, :])
        return t_o, pos_o, n_o

    def _intersect(self, direction: Array, source: Array, min_t: Array, max_t: Array) -> Tuple[Array, Array, Array]:
        """
        Args:
            direction(Array): [3, N]，**物体坐标系下**光线的方向向量
            source(Array): [3, N]，**物体坐标系下**光线起点的位置
            min_t(Array): [N]，**物体坐标系下**直线与物体交点的最小距离，小于该距离说明交点在近裁剪平面前方
            max_t(Array)：[N]，**物体坐标系下**直线与物体交点的最大距离，大于该距离说明交点在近裁剪平面后方
        Returns:
            t(Array): [N]，**物体坐标系下**交点与光线起点的距离，若无交点则为np.inf
            pos(Array): [3, N]，**物体坐标系下**交点位置，若无交点则为[np.inf, np.inf, np.inf]
            n(Array): [3, N]，**物体坐标系下**交点法线（可以不归一化，由intersect归一化）
        """
        raise NotImplementedError()