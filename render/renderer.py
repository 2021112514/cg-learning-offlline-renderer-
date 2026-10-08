from device import np, Array
from ray import Ray
from scene import Scene
from settings.settings import Settings
from transform import normalize, build_coordinate_system


class Renderer:
    def __init__(self, scene: Scene, settings: Settings):
        self.scene = scene
        self.settings = settings
        

    def render(self) -> Array:
        color = self._compute_radiance(self.scene.camera)
        if self.scene.camera.sample_times > 0:
            color = color.reshape((3, self.scene.camera.sample_times, -1)).mean(axis=1)
        color = np.array(np.clip(color * 255, a_min=0.0, a_max=255.0), dtype=np.uint8)
        return color.reshape((3, self.scene.camera.H, self.scene.camera.W)).transpose(2, 1, 0)

    def _compute_radiance(self, ray: Ray, depth: int=0) -> Array:
        color = np.zeros_like(ray.direction)
        t, pos, n, object_index = self.scene.intersect(ray)
        material_index = np.where(object_index >= 0, self.scene.object_material_index_map[object_index.clip(min=0, max=None)], -1)
        num_material = int(material_index.max() + 1)
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
        min_t = np.full(pos_selected.shape[-1], 1e-6)
        max_t_inf = np.full_like(min_t, np.inf)
        material_index_selected = material_index[selected]
        for light in self.scene.lights:
            radiance, source = light.get_radiance(pos_selected) # [3, N_selected]
            shadow_ray_direction = normalize(source - pos_selected)
            max_t = light.get_distance(pos_selected)
            shadow_ray = Ray(
                direction=shadow_ray_direction,
                pos=pos_selected,
                min_t=min_t,
                max_t=max_t
            )
            t_, _, _, _ = self.scene.intersect(shadow_ray)
            not_blocked = t_ == np.inf # [N_selected]
            not_blocked_global = np.full(N, False)
            not_blocked_global[idx_selected] = not_blocked
            cos = (n_selected * normalize(source - pos_selected)).sum(axis=0) # [N_selected]
            # 以法线为z轴的坐标系中，入射光线的方向，用于计算BRDF
            wi_local = (coord_transform_not_inf.transpose((1, 2, 0)) * shadow_ray_direction[None, :, :]).sum(axis=1)
            brdf = np.full((3, int(not_blocked.sum())), 0.0)
            material_index_not_blocked = material_index[not_blocked_global]
            for material_id in range(num_material):
                material = self.scene.materials[material_id]
                mask = material_index_not_blocked == material_id
                brdf[:, mask] = material.brdf(wo_local[:, not_blocked][:, mask], wi_local[:, not_blocked][:, mask])
            color[:, not_blocked_global] += radiance[:, not_blocked] * np.clip(cos[not_blocked], a_min=0.0, a_max=None) * brdf
        if depth == self.settings.render.max_reflection_time:
            return color / factor
        # 间接光照
        color_sum = np.full((3, N_selected), 0.0)
        for _ in range(self.settings.render.monte_carlo_integration_sample_times):
            theta = np.random.rand(N_selected) * np.pi * 0.5
            phi = np.random.rand(N_selected) * np.pi * 2
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            wi = np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)
            brdf = np.full((3, N_selected), 0.0)
            for material_id in range(num_material):
                material = self.scene.materials[material_id]
                mask = material_index_selected == material_id
                brdf[:, mask] = material.brdf(wo_local[:, mask], wi[:, mask])
            direction_wi_global = (coord_transform_inv.transpose((1, 2, 0)) * wi).sum(axis=1)
            ray = Ray(
                direction=direction_wi_global,
                pos=pos_selected,
                min_t=min_t,
                max_t=max_t_inf
            )
            color_ = brdf * self._compute_radiance(ray, depth + 1) * cos_theta * sin_theta
            color_sum += color_
        color[:, selected] = color[:, selected] + (np.pi ** 2 * color_sum / self.settings.render.monte_carlo_integration_sample_times)
        return color / factor