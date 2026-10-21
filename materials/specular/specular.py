from typing import Optional
from materials import Material


class Specular(Material):
    def __init__(self, name: str, albedo_map: Optional[str] = None, normal_map: Optional[str] = None, bump_scale: float = 1):
        super().__init__(name, albedo_map, normal_map, bump_scale)