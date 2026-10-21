from typing import Optional
from materials import Material


class Specular(Material):
    def __init__(self, name: str, albedo_map: Optional[str] = None):
        super().__init__(name, albedo_map)