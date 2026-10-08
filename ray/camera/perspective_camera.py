from device import np, Array
from typing import Tuple
from transform import view_transform, projection, viewport, normalize, normalize_homogeneous
from ..ray import Ray
from parser import tag


@tag('Camera')
class PerspectiveCamera(Ray):
    def __init__(
        self,
        position: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float],
        fov: float,
        near_z: float,
        far_z: float
    ):
        from run import settings
        self.W = settings.screen.w
        self.H = settings.screen.h
        alpha = self.W / self.H
        p = np.array(position)
        l = normalize(np.array(target) - p)
        r = np.cross(np.array(up), l)
        assert (r ** 2).sum() > 1e-8
        r = normalize(r)
        u = np.cross(l, r)
        _, Mv_inv = view_transform(r, u, l, p)
        _, Mp_inv = projection(fov, alpha, near_z, far_z)
        _, Mvp_inv = viewport(self.W, self.H)
        matrix = Mv_inv @ Mp_inv @ Mvp_inv
        X, Y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        points = np.stack([X, Y], axis=-1).reshape(-1, 2).T + np.array([[0.5], [0.5]])

        # SSAA
        self.sample_times = settings.render.ssaa_sample_times
        if self.sample_times > 0:
            sample = np.random.rand(self.sample_times, 2)[..., None] - 0.5
            points = (sample + points).transpose((1, 0, 2)).reshape(2, -1)

        points_expanded = np.concat([points, np.zeros((1, points.shape[-1])), np.ones((1, points.shape[-1]))], axis=0)

        # 屏幕坐标 (x, y, 0, 1) 映射到世界坐标
        # 由投影矩阵的逆矩阵可知，当屏幕坐标后两个分量分别取z=0,w=1时，映射后的点在近裁剪平面上：
        # 视口变换的逆矩阵不改变z和w分量，投影变换的逆使得z=1,w=1/n，规范化后z=n,w=1（摄像机坐标下），故在近裁剪平面上
        world_points = normalize_homogeneous(matrix @ points_expanded)[:3, :]
        direction = world_points - p[:, None]
        norm = np.sqrt((direction ** 2).sum(axis=0))
        self.direction: Array = direction / norm # [3, N]

        # norm为摄像机到直线与近裁剪平面的交点的距离，据此计算直线的max_t和min_t
        self.min_t = norm # [N]
        self.max_t = norm * far_z / near_z # [N]

        self.pos = p[:, None] # [3, 1]