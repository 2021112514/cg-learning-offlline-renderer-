import math
from tqdm import tqdm
from device import np, Array
from ray import Ray, PerspectiveCamera
from scene import Scene
from settings.settings import Settings
from transform import normalize, build_coordinate_system, get_refractor_vector
from material import DielectricSpecular


class Renderer:
    def __init__(self, scene: Scene, settings: Settings):
        self.scene = scene
        self.settings = settings
        

    def render(self) -> Array:
        # 按SSAA采样对点进行分片，减少显存压力
        with tqdm(total=self.settings.render.split) as pbar:
            pbar.set_description('Rendering slices')
            color_sum = np.zeros((self.scene.camera.W, self.scene.camera.H, 3))
            camera = self.scene.camera
            color = self._compute_radiance(camera)
            pbar.update(1)
            if camera.sample_times > 0:
                color = color.reshape((3, camera.sample_times, -1)).sum(axis=1)
            color_sum += color.reshape((3, self.scene.camera.H, self.scene.camera.W)).transpose(2, 1, 0)
            max_sampling_times = math.ceil(self.settings.render.ssaa_sampling_times / self.settings.render.split)
            sampling_times_used = max_sampling_times
            for _ in range(1, self.settings.render.split):
                times = min(self.settings.render.ssaa_sampling_times - sampling_times_used, max_sampling_times)
                camera = PerspectiveCamera(
                    position=self.scene.camera.position,
                    target=self.scene.camera.target,
                    up=self.scene.camera.up,
                    fov=self.scene.camera.fov,
                    near_z=self.scene.camera.near_z,
                    far_z=self.scene.camera.far_z,
                    sample_times=times
                )
                sampling_times_used += times
                color = self._compute_radiance(camera)
                pbar.update(1)
                if camera.sample_times > 0:
                    color = color.reshape((3, camera.sample_times, -1)).sum(axis=1)
                color_sum += color.reshape((3, self.scene.camera.H, self.scene.camera.W)).transpose(2, 1, 0)
        return np.array(np.clip((color_sum / self.settings.render.ssaa_sampling_times) * 255, a_min=0.0, a_max=255.0), dtype=np.uint8)

    def _compute_radiance(self, ray: Ray, depth: int=0) -> Array:
        color = np.zeros_like(ray.direction)
        t, pos, n, object_index = self.scene.intersect(ray)
        N = t.shape[0]
        not_inf = t < np.inf
        # 俄罗斯轮盘算法实现路径追踪
        if depth <= self.settings.render.min_reflection_time:
            selected = not_inf
            factor = 1
        else:
            selected = np.logical_and(np.random.rand(N) <= self.settings.render.reflection_probably, not_inf)
            factor = self.settings.render.reflection_probably
        N_selected = int(selected.sum())
        if N_selected == 0:
            return color
        # 没有被选中的点返回0（而不是直接光照）
        idx_selected = np.arange(N)[selected]
        n_selected = n[:, selected]
        coord_transform_not_inf, coord_transform_inv = build_coordinate_system(n_selected) # [N_selected, 3, 3]
        # 以法线为z轴的坐标系中，出射光线的方向，用于计算BRDF
        # 暂不考虑具有各向异性的材质（建系时不能仅指定法线）
        wo_local = (coord_transform_not_inf.transpose((1, 2, 0)) * -ray.direction[None, :, selected]).sum(axis=1) # [3, N_selected]
        pos_selected = pos[:, selected]
        min_t = np.array([1e-6])
        max_t_inf = np.full_like(min_t, np.inf)
        object_index_selected = object_index[selected]
        material_index_selected = self.scene.object_material_index_map[object_index_selected]
        num_material = int(material_index_selected.max() + 1)
        is_specular = self.scene.object_is_specular[object_index_selected]
        not_specular = np.logical_not(is_specular)
        pos_not_specular = pos_selected[:, not_specular]
        material_index_not_specular = material_index_selected[not_specular]
        material_index_specular = material_index_selected[is_specular]
        N_not_specular = pos_not_specular.shape[-1]
        N_specular = int(is_specular.sum())
        for light in self.scene.lights:
            radiance, source = light.get_radiance(pos_not_specular) # [3, N_not_specular]
            shadow_ray_direction = normalize(source - pos_not_specular)
            max_t = light.get_distance(pos_not_specular)
            shadow_ray = Ray(
                direction=shadow_ray_direction,
                pos=pos_not_specular,
                min_t=np.broadcast_to(min_t, N_not_specular),
                max_t=max_t
            )
            t_, _, _, _ = self.scene.intersect(shadow_ray)
            not_blocked = t_ == np.inf # [N_not_specular]
            not_blocked_global = np.full(N, False)
            not_blocked_global[idx_selected[not_specular]] = not_blocked
            cos = (n_selected[:, not_specular] * normalize(source - pos_not_specular)).sum(axis=0) # [N_not_specular]
            # 以法线为z轴的坐标系中，入射光线的方向，用于计算BRDF
            wi_local = (coord_transform_not_inf[not_specular, :, :].transpose((1, 2, 0)) * shadow_ray_direction[None, :, :]).sum(axis=1)
            brdf = np.full((3, int(not_blocked.sum())), 0.0)
            material_index_not_blocked = material_index_not_specular[not_blocked]
            for material_id in range(num_material):
                material = self.scene.materials[material_id]
                mask = material_index_not_blocked == material_id
                brdf[:, mask] = material.brdf(wo_local[:, not_specular][:, not_blocked][:, mask], wi_local[:, not_blocked][:, mask])
            color[:, not_blocked_global] += radiance[:, not_blocked] * np.clip(cos[not_blocked], a_min=0.0, a_max=None) * brdf
        if depth == self.settings.render.max_reflection_time:
            return color / factor
        # 间接光照
        color_sum = np.full((3, N_not_specular), 0.0)
        # 非镜面材质，采用 Monte Carlo 积分求解渲染方程
        for _ in range(self.settings.render.monte_carlo_integration_sampling_times):
            wi = np.zeros((3, N_not_specular))
            pdf = np.ones(N_not_specular)
            brdf = np.full((3, N_not_specular), 0.0)
            for material_id in range(num_material):
                material = self.scene.materials[material_id]
                mask = material_index_not_specular == material_id
                if int(np.sum(mask)) > 0:
                    sampled_wi, sampled_pdf = material.sample_wi(wo_local[:, not_specular][:, mask])
                    wi[:, mask] = sampled_wi
                    pdf[mask] = sampled_pdf
                    brdf[:, mask] = material.brdf(wo_local[:, not_specular][:, mask], wi[:, mask])
            direction_wi_global = (coord_transform_inv[not_specular, :, :].transpose((1, 2, 0)) * wi).sum(axis=1)
            ray = Ray(
                direction=direction_wi_global,
                pos=pos_not_specular,
                min_t=np.broadcast_to(min_t, N_not_specular),
                max_t=np.broadcast_to(max_t_inf, N_not_specular)
            )
            color_ = brdf * self._compute_radiance(ray, depth + 1) * wi[-1, :] / pdf
            color_sum += color_
        color[:, idx_selected[not_specular]] += color_sum / self.settings.render.monte_carlo_integration_sampling_times
        # 镜面材质，渲染方程的曲面积分退化为单入射方向
        wi = wo_local[:, is_specular] * np.array([[-1.0], [-1.0], [1.0]])
        brdf = np.zeros((3, N_specular))
        for material_id in range(num_material):
            material = self.scene.materials[material_id]
            mask = material_index_specular == material_id
            brdf[:, mask] = material.brdf(wo_local[:, is_specular][:, mask], wi[:, mask])
        direction_wi_global = np.sum(coord_transform_inv[is_specular, :, :].transpose((1, 2, 0)) * wi, axis=1)
        ray = Ray(
            direction=direction_wi_global,
            pos=pos_selected[:, is_specular],
            min_t=np.broadcast_to(min_t, N_specular),
            max_t=np.broadcast_to(max_t_inf, N_specular)
        )
        cos_theta = np.abs(wi[-1, :])
        color_ = brdf * self._compute_radiance(ray, depth + 1) * cos_theta
        color[:, idx_selected[is_specular]] += color_
        # 折射
        is_refractable = self.scene.object_is_refractable[object_index_selected]
        N_refractable = int(np.sum(is_refractable))
        if N_refractable > 0:
            btdf = np.zeros((3, N_refractable))
            material_index_refractable = material_index_selected[is_refractable]
            wi = wo_local[:, is_refractable]
            arange_refractable = np.arange(material_index_refractable.shape[0])
            idx_refractable_in_selected = np.arange(idx_selected.shape[0])[is_refractable]
            not_full_refractor = np.full_like(idx_refractable_in_selected, False, dtype=bool)
            wt_local = np.zeros_like(wi)
            for material_id in range(num_material):
                material = self.scene.materials[material_id]
                mask = arange_refractable[material_index_refractable == material_id]
                if mask.shape[0] > 0:
                    assert isinstance(material, DielectricSpecular)
                    eta = np.where(wi[-1, mask] > 0, 1 / material.eta, material.eta)
                    wt, refractor = get_refractor_vector(wi[:, mask], eta)
                    wt_local[:, mask[refractor]] = wt
                    not_full_refractor[mask[refractor]] = True
                    btdf[:, mask[refractor]] = material.btdf(wi[:, mask[refractor]], wt) # BTDF调用时的参数顺序问题见教程第11讲46分处
            idx_not_full_refractor_in_selected = idx_refractable_in_selected[not_full_refractor]
            wt_global = np.sum(coord_transform_inv[idx_not_full_refractor_in_selected].transpose((1, 2, 0)) * wt_local[:, not_full_refractor], axis=1)
            if idx_not_full_refractor_in_selected.shape[0] > 0:
                ray = Ray(
                    direction=wt_global,
                    pos=pos_selected[:, idx_not_full_refractor_in_selected],
                    min_t=np.broadcast_to(min_t, idx_not_full_refractor_in_selected.shape[0]),
                    max_t=np.broadcast_to(max_t_inf, idx_not_full_refractor_in_selected.shape[0])
                )
                cos_theta = wt_local[:, not_full_refractor][-1, :]
                color_ = btdf[:, not_full_refractor] * self._compute_radiance(ray, depth + 1) * np.abs(cos_theta)
                color[:, idx_selected[idx_not_full_refractor_in_selected]] += color_
        return color / factor