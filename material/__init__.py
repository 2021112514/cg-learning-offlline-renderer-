from .material import Material
from .lambert import Lambert
from parser import tag_constructor


@tag_constructor('Material')
class MaterialConstructor:
    @staticmethod
    def construct_tag(type: str, **kwargs):
        try:
            cls = globals()[type]
            if not issubclass(cls, Material):
                raise TypeError(f"Unknown Material {type}")
            return cls(**kwargs)
        except KeyError:
            raise KeyError(f"Unknown Material: {type}")